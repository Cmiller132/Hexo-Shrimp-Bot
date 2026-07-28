"""The machine under the run: GPU, process, and host counters, sampled.

A metrics row says how many samples an iteration produced; it says nothing
about whether the card was saturated, throttling, or waiting on the host —
and every throughput question the run plan asks is one of those. So a
background thread reads the counters on a fixed period and the run drains
its aggregate once per iteration, alongside the iteration's own numbers.

The thread is deliberately inert with respect to training: it touches no
model, no RNG, and no database, and it holds nothing the training threads
want. Its one interaction is the drain, which is a list swap under a lock.

Sensor failures are kept, not swallowed. A read that raises stops the
sampler and the exception is re-raised on the training thread at the next
drain, so a run whose hardware trace has quietly stopped is not a thing
that can happen.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import torch

# NVML reports power in milliwatts and memory in bytes; both are converted
# at the sample so every stored column is in the unit its name says.
_MILLIWATT = 1e-3


class HardwareSampler:
    """Counters on a fixed period, aggregated per drain.

    ``gpu_index`` is the NVML device to watch, or ``None`` for a run with no
    GPU at all — a CPU run then reports the process and host columns and
    leaves the GPU ones empty, which is what "there is no GPU" looks like.
    """

    def __init__(self, gpu_index: int | None, period: float = 1.0):
        import psutil

        self.period = period
        self._gpu_index = gpu_index
        self._proc = psutil.Process()
        self._psutil = psutil
        self._proc.cpu_percent(None)  # prime the interval this reads against
        self._handle = None
        self._nvml = None
        if gpu_index is not None:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._lock = threading.Lock()
        self._samples: list[dict] = []
        self._failure: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="hardware-sampler", daemon=True
        )
        self._thread.start()

    def _sample(self) -> dict:
        s = {
            "cpu_percent": self._proc.cpu_percent(None),
            "threads": self._proc.num_threads(),
            "rss": self._proc.memory_info().rss,
            "sys_ram_used": self._psutil.virtual_memory().used,
        }
        if self._handle is not None:
            nvml, h = self._nvml, self._handle
            s["gpu_util"] = float(nvml.nvmlDeviceGetUtilizationRates(h).gpu)
            s["gpu_power_w"] = nvml.nvmlDeviceGetPowerUsage(h) * _MILLIWATT
            s["gpu_mem_used"] = nvml.nvmlDeviceGetMemoryInfo(h).used
            s["gpu_temp"] = float(
                nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
            )
        return s

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            try:
                s = self._sample()
            except BaseException as exc:  # re-raised on the next drain
                with self._lock:
                    self._failure = exc
                return
            with self._lock:
                self._samples.append(s)

    def drain(self) -> dict:
        """One iteration's hardware columns: means and maxima over the
        samples since the last drain, plus torch's own allocator peaks.

        An iteration shorter than the sample period would otherwise report
        nothing, so an empty buffer is filled by sampling inline — the read
        is a few counter lookups, and a column that is always present is
        worth more to a threshold query than one that is usually present.
        """
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("the hardware sampler stopped") from self._failure
            samples, self._samples = self._samples, []
        if not samples:
            samples = [self._sample()]

        out: dict = {"hw_samples": len(samples)}
        for key in samples[0]:
            values = [s[key] for s in samples]
            out[f"{key}_mean"] = sum(values) / len(values)
            out[f"{key}_max"] = max(values)
        if self._gpu_index is not None:
            # Peak-since-last-drain: reset here so the column is the
            # iteration's peak rather than the process's.
            out["torch_alloc_max"] = torch.cuda.max_memory_allocated()
            out["torch_reserved_max"] = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        return out

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5 * self.period)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
            self._nvml, self._handle = None, None


@contextmanager
def hardware_sampler(device: str, period: float = 1.0):
    """The sampler for a run on ``device``, shut down with the run.

    torch's device index is NVML's only while one GPU is visible; the two
    orderings are independent, and on a multi-GPU host this would quietly
    trace the wrong card. Rather than guess, it stops — matching the two by
    UUID is the fix, when there is a host to test it on.
    """
    index = None
    if device.startswith("cuda"):
        if torch.cuda.device_count() > 1:
            raise RuntimeError(
                f"{torch.cuda.device_count()} visible GPUs: torch's device index is "
                "not NVML's, so the hardware trace would name the wrong card — "
                "pin one with CUDA_VISIBLE_DEVICES"
            )
        index = torch.cuda.current_device()
    sampler = HardwareSampler(index, period)
    try:
        yield sampler
    finally:
        sampler.close()


def _main() -> None:
    """A few seconds of the trace, for checking the sensors exist."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--period", type=float, default=0.5)
    args = ap.parse_args()
    with hardware_sampler(args.device, args.period) as hw:
        time.sleep(args.seconds)
        print(json.dumps(hw.drain(), indent=2))


if __name__ == "__main__":
    _main()
