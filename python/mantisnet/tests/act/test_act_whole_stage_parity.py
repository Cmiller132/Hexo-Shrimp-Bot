"""End-to-end parity for ACT's one outer ``torch.compile`` boundary.

Section 36 permits an optimized path only when random weights and a real
checkpoint are held against a reference implementation.  ACT's optimized path
is now the same shape as production MantisNet's: the complete model is ordinary
traceable torch code inside one ``torch.compile(dynamic=True)`` callable, while
only the four hand-written Triton families remain opaque registered operators.

The compiled and eager models are exact state-dict clones over real stack-939
prefixes.  In fp32 the only expected disagreement is compiler reassociation, so
every forward tensor and every named parameter gradient is gated at 2e-4
relative.  Mathematically zero gradients (the latent key biases) are measured
against the model-wide gradient scale rather than divided by zero.  The bf16
arm reports the compiled bf16 run against a compiled fp32 anchor and gates
finiteness plus successful hand-kernel dispatch.  A separate null control
requires compiled fp32 outputs and gradients to repeat bitwise.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

import pytest
import torch
from torch import Tensor

from mantisnet.models.mantis_act import (
    class_embedding,
    latent_attention,
    post_rows,
    segment_message,
)
from mantisnet.models.mantis_act.builder import build
from mantisnet.models.mantis_act.config import PRESETS, MantisACTConfig
from mantisnet.models.mantis_act.model import ACTOutput, MantisACT
from mantisnet.models.mantis_act.packed import PackedACTBatch, collate

from .test_act_d6 import randomise_
from .test_act_numerics import position


SEED = 20260809
FULL = PRESETS["full_act_v4"]
PLIES = (21, 61, 121, 161)
BATCH_SIZES = (8, 16)
MASS_FLOOR = 0.2
FP32_RELATIVE_TOLERANCE = 2e-4
ZERO_GRADIENT_ULPS = 32

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="outer-compiled parity must exercise the registered CUDA kernels",
)

ModelFactory = Callable[[], MantisACT]
ModelForward = Callable[[MantisACT, PackedACTBatch], ACTOutput]


@dataclass(frozen=True)
class Snapshot:
    outputs: dict[str, Tensor]
    gradients: dict[str, Tensor]


def _real_batch(cfg: MantisACTConfig, ply: int, batch_size: int) -> PackedACTBatch:
    """Alternate the two embedded real games to the requested exact batch size."""
    representatives = [build(position(game, ply), cfg) for game in (0, 1)]
    graphs = [
        representatives[index % len(representatives)] for index in range(batch_size)
    ]
    return collate(graphs, cfg)


@pytest.fixture(scope="module")
def random_state() -> dict[str, Tensor]:
    torch.manual_seed(SEED)
    model = randomise_(MantisACT(FULL), SEED)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _factory_from_state(
    cfg: MantisACTConfig, state: Mapping[str, Tensor]
) -> ModelFactory:
    def factory() -> MantisACT:
        model = MantisACT(cfg)
        model.load_state_dict(state, strict=True)
        return model.eval()

    return factory


def _outer_forward(model: MantisACT, batch: PackedACTBatch) -> ACTOutput:
    """The sole compile boundary, matching the production whole-model seam."""
    return model(batch, MASS_FLOOR)


@pytest.fixture(scope="module")
def compiled_forward() -> ModelForward:
    """One unbroken shape-polymorphic graph shared by the parity battery."""
    return torch.compile(_outer_forward, dynamic=True, fullgraph=True)


def _output_tensors(output: ACTOutput) -> dict[str, Tensor]:
    values = {
        "policy_logits": output.policy_logits,
        "critic_logits": output.critic_logits,
        "q_value": output.q_value,
        "q_score": output.q_score,
        "legal_offsets": output.legal_offsets,
    }
    values.update({f"aux.{name}": value for name, value in sorted(output.aux.items())})
    return values


def _loss(output: ACTOutput) -> Tensor:
    """A bounded scalar to which every floating output contributes."""
    terms = [
        value.float().square().mean()
        for value in _output_tensors(output).values()
        if value.is_floating_point() and value.numel()
    ]
    if not terms:
        raise AssertionError("the ACT output exposed no floating tensor")
    return sum(terms)


_HAND_KERNELS = {
    "segment_message": (
        segment_message,
        ("_launch_forward", "_launch_backward"),
    ),
    "latent_attention": (
        latent_attention,
        (
            "_launch_read",
            "_launch_read_backward",
            "_launch_broadcast",
            "_launch_broadcast_backward",
        ),
    ),
    "post_rows": (
        post_rows,
        (
            "_launch_gather",
            "_launch_gather_backward",
            "_launch_row_gate",
            "_launch_row_gate_backward",
        ),
    ),
}


def _clear_hand_kernel_failures() -> None:
    for module, _launches in (*_HAND_KERNELS.values(), (class_embedding, ())):
        for name in (
            "_FAILED_SHAPES",
            "_FAILED_FORWARD_SHAPES",
            "_FAILED_BACKWARD_SHAPES",
        ):
            cache = getattr(module, name, None)
            if cache is not None:
                cache.clear()


def _assert_no_hand_kernel_failures() -> None:
    failures = {}
    for family, (module, _launches) in {
        **_HAND_KERNELS,
        "class_embedding": (class_embedding, ()),
    }.items():
        for name in (
            "_FAILED_SHAPES",
            "_FAILED_FORWARD_SHAPES",
            "_FAILED_BACKWARD_SHAPES",
        ):
            cache = getattr(module, name, None)
            if cache:
                failures[f"{family}.{name}"] = dict(cache)
    assert not failures, failures


@contextmanager
def _count_hand_kernel_launches():
    """Prove that every eligible default call reaches its real Triton helper."""
    originals = {}
    counts = {}
    for family, (module, launches) in _HAND_KERNELS.items():
        counts[family] = {"eligible": 0, **{name: 0 for name in launches}}
        originals[(family, "_supported")] = module._supported
        for name in launches:
            originals[(family, name)] = getattr(module, name)

    # Class lookup's forward is ordinary torch.  Its class-sorted ordered
    # backward remains the required hand kernel, so hold backward eligibility
    # and successful launch against each other exactly like the other families.
    counts["class_embedding"] = {"eligible": 0, "_launch_backward": 0}
    originals[("class_embedding", "_supported")] = class_embedding._supported
    originals[("class_embedding", "_launch_backward")] = (
        class_embedding._launch_backward
    )

    def count_supported(family, original):
        def wrapped(*args, **kwargs):
            accepted = original(*args, **kwargs)
            counts[family]["eligible"] += int(bool(accepted))
            return accepted

        return wrapped

    def count_launch(family, name, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            counts[family][name] += 1
            return result

        return wrapped

    for family, (module, launches) in _HAND_KERNELS.items():
        module._supported = count_supported(
            family, originals[(family, "_supported")]
        )
        for name in launches:
            setattr(
                module,
                name,
                count_launch(family, name, originals[(family, name)]),
            )
    class_embedding._launch_backward = count_launch(
        "class_embedding",
        "_launch_backward",
        originals[("class_embedding", "_launch_backward")],
    )
    class_embedding._supported = count_supported(
        "class_embedding",
        originals[("class_embedding", "_supported")],
    )

    try:
        yield counts
    finally:
        for family, (module, launches) in _HAND_KERNELS.items():
            module._supported = originals[(family, "_supported")]
            for name in launches:
                setattr(module, name, originals[(family, name)])
        class_embedding._launch_backward = originals[
            ("class_embedding", "_launch_backward")
        ]
        class_embedding._supported = originals[("class_embedding", "_supported")]


def _assert_successful_hand_kernel_launches(counts) -> None:
    for family, (_module, launches) in _HAND_KERNELS.items():
        launched = sum(counts[family][name] for name in launches)
        assert counts[family]["eligible"] == launched and launched > 0, counts
        assert all(counts[family][name] > 0 for name in launches), counts
    assert (
        counts["class_embedding"]["eligible"]
        == counts["class_embedding"]["_launch_backward"]
        > 0
    ), counts
    _assert_no_hand_kernel_failures()


def _capture(
    factory: ModelFactory,
    batch: PackedACTBatch,
    forward: ModelForward,
    *,
    bf16: bool = False,
) -> Snapshot:
    model = factory().to("cuda").eval()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = forward(model, batch)
        loss = _loss(output)
    loss.backward()
    torch.cuda.synchronize()

    outputs = {
        name: value.detach().cpu().clone()
        for name, value in _output_tensors(output).items()
    }
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    assert not missing, f"parameters disconnected from the whole-model loss: {missing}"
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    for family, values in (("output", outputs), ("gradient", gradients)):
        bad = [
            name
            for name, value in values.items()
            if value.is_floating_point() and not torch.isfinite(value).all()
        ]
        assert not bad, f"non-finite {family} tensors: {bad}"

    del loss, output, model
    torch.cuda.empty_cache()
    return Snapshot(outputs, gradients)


def _eager_capture(factory: ModelFactory, batch: PackedACTBatch) -> Snapshot:
    return _capture(factory, batch, _outer_forward)


def _compiled_capture(
    factory: ModelFactory,
    batch: PackedACTBatch,
    compiled_forward: ModelForward,
    *,
    bf16: bool = False,
) -> Snapshot:
    _clear_hand_kernel_failures()
    with _count_hand_kernel_launches() as counts:
        snapshot = _capture(factory, batch, compiled_forward, bf16=bf16)
    _assert_successful_hand_kernel_launches(counts)
    return snapshot


def _magnitude(value: Tensor) -> float:
    if not value.numel():
        return 0.0
    return float(value.double().abs().max())


def _relative(actual: Tensor, expected: Tensor, denominator: float) -> float:
    if not actual.numel():
        return 0.0
    difference = float((actual.double() - expected.double()).abs().max())
    if denominator == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / denominator


def _parity_metrics(
    actual: Snapshot, expected: Snapshot
) -> tuple[dict[str, float], dict[str, float]]:
    assert set(actual.outputs) == set(expected.outputs)
    assert set(actual.gradients) == set(expected.gradients)

    output_metrics: dict[str, float] = {}
    for name, reference in expected.outputs.items():
        value = actual.outputs[name]
        assert value.shape == reference.shape, name
        if not reference.is_floating_point():
            assert torch.equal(value, reference), name
            continue
        output_metrics[name] = _relative(value, reference, _magnitude(reference))

    global_gradient_scale = max(
        (_magnitude(value) for value in expected.gradients.values()), default=0.0
    )
    assert global_gradient_scale > 0.0, "the whole-model loss produced only zero grads"
    numerical_zero = (
        ZERO_GRADIENT_ULPS * torch.finfo(torch.float32).eps * global_gradient_scale
    )
    gradient_metrics: dict[str, float] = {}
    for name, reference in expected.gradients.items():
        value = actual.gradients[name]
        assert value.shape == reference.shape, name
        own_scale = _magnitude(reference)
        denominator = (
            global_gradient_scale if own_scale <= numerical_zero else own_scale
        )
        gradient_metrics[name] = _relative(value, reference, denominator)
    return output_metrics, gradient_metrics


def _assert_fp32_parity(actual: Snapshot, expected: Snapshot) -> None:
    output_metrics, gradient_metrics = _parity_metrics(actual, expected)
    failures = [
        (f"output.{name}", relative)
        for name, relative in output_metrics.items()
        if relative > FP32_RELATIVE_TOLERANCE
    ]
    failures.extend(
        (f"parameter.{name}", relative)
        for name, relative in gradient_metrics.items()
        if relative > FP32_RELATIVE_TOLERANCE
    )
    failures.sort(key=lambda item: item[1], reverse=True)
    assert not failures, "reassociation exceeded 2e-4 relative:\n" + "\n".join(
        f"{name}: {relative:.6e}" for name, relative in failures[:20]
    )


def _assert_bitwise(actual: Snapshot, expected: Snapshot) -> None:
    assert set(actual.outputs) == set(expected.outputs)
    assert set(actual.gradients) == set(expected.gradients)
    failures = [
        f"output.{name}"
        for name, reference in expected.outputs.items()
        if not torch.equal(actual.outputs[name], reference)
    ]
    failures.extend(
        f"parameter.{name}"
        for name, reference in expected.gradients.items()
        if not torch.equal(actual.gradients[name], reference)
    )
    assert not failures, "compiled fp32 was not bitwise deterministic: " + ", ".join(
        failures[:20]
    )


def _run_parity(
    factory: ModelFactory,
    cfg: MantisACTConfig,
    ply: int,
    batch_size: int,
    compiled_forward: ModelForward,
) -> None:
    batch = _real_batch(cfg, ply, batch_size).to("cuda")
    expected = _eager_capture(factory, batch)
    actual = _compiled_capture(factory, batch, compiled_forward)
    _assert_fp32_parity(actual, expected)
    del batch, expected, actual
    torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.parametrize("ply", PLIES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_random_weight_fp32_outer_compiled_model_matches_eager(
    random_state: Mapping[str, Tensor],
    compiled_forward: ModelForward,
    ply: int,
    batch_size: int,
) -> None:
    """Only fp32 reassociation may separate the two exact state-dict clones."""
    _run_parity(
        _factory_from_state(FULL, random_state),
        FULL,
        ply,
        batch_size,
        compiled_forward,
    )


@requires_cuda
def test_real_checkpoint_fp32_outer_compiled_model_matches_eager(
    compiled_forward: ModelForward,
) -> None:
    """Run the section 36 checkpoint oracle when its owner supplies the artifact."""
    raw_path = os.environ.get("MANTIS_ACT_CHECKPOINT", "").strip()
    if not raw_path:
        pytest.skip(
            "MANTIS_ACT_CHECKPOINT is unset; set it to a full_act_v4 checkpoint "
            "to run the required real-checkpoint outer-compile oracle"
        )
    path = Path(raw_path)
    assert path.is_file(), f"MANTIS_ACT_CHECKPOINT does not name a file: {path}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    probe = MantisACT.from_checkpoint(payload).eval()
    assert probe.cfg == FULL, (
        "the whole-model checkpoint oracle requires full_act_v4, got "
        f"{probe.cfg}"
    )

    def factory() -> MantisACT:
        return MantisACT.from_checkpoint(payload).eval()

    del probe
    _run_parity(factory, FULL, 161, 8, compiled_forward)


@requires_cuda
def test_bf16_error_is_reported_against_compiled_fp32_anchor_not_itself(
    random_state: Mapping[str, Tensor],
    compiled_forward: ModelForward,
    record_property,
) -> None:
    """Document bf16 drift; only finiteness and real kernel launches are gates."""
    factory = _factory_from_state(FULL, random_state)
    batch = _real_batch(FULL, ply=161, batch_size=8).to("cuda")
    anchor = _compiled_capture(factory, batch, compiled_forward)
    bf16 = _compiled_capture(factory, batch, compiled_forward, bf16=True)
    output_metrics, gradient_metrics = _parity_metrics(bf16, anchor)
    worst_output = max(output_metrics.items(), key=lambda item: item[1])
    worst_gradient = max(gradient_metrics.items(), key=lambda item: item[1])
    record_property("bf16_fp32_worst_output", worst_output[1])
    record_property("bf16_fp32_worst_output_name", worst_output[0])
    record_property("bf16_fp32_worst_gradient", worst_gradient[1])
    record_property("bf16_fp32_worst_gradient_name", worst_gradient[0])
    del batch, anchor, bf16
    torch.cuda.empty_cache()


@requires_cuda
def test_outer_compiled_fp32_outputs_and_gradients_are_bitwise_deterministic(
    random_state: Mapping[str, Tensor], compiled_forward: ModelForward
) -> None:
    """The compiler may reassociate against eager, but repeated runs are exact."""
    factory = _factory_from_state(FULL, random_state)
    batch = _real_batch(FULL, ply=161, batch_size=8).to("cuda")
    first = _compiled_capture(factory, batch, compiled_forward)
    second = _compiled_capture(factory, batch, compiled_forward)
    _assert_bitwise(second, first)
    del batch, first, second
    torch.cuda.empty_cache()
