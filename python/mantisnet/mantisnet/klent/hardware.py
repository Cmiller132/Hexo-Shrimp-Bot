"""Sample GPU, process, and host counters for iteration telemetry.

A background thread samples on a fixed period. ``drain`` returns means and
maxima since the preceding drain without touching model or RNG state. A sensor
failure stops sampling and is raised by the next drain.
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
    GPU. A CPU run reports process and host columns and leaves GPU columns
    empty.
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
            except BaseException as exc:  # Surface failures at the next drain.
                with self._lock:
                    self._failure = exc
                return
            with self._lock:
                self._samples.append(s)

    def drain(self) -> dict:
        """One iteration's hardware columns: means and maxima over the
        samples since the last drain, plus torch's own allocator peaks.

        If no periodic sample is available, this method samples inline so
        every drain contains one or more observations.
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

    CUDA sampling requires at most one visible GPU because torch and NVML
    device-index orderings are not assumed to match.
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
    """Sample hardware counters for a bounded interval and print one drain."""
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
