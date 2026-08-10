"""End-to-end parity for the registered whole-stage ACT operators.

Section 36 permits an optimized path only when random weights and a real
checkpoint are held against a reference implementation.  This file restores
the literal pre-fusion compositions at their Python dispatch seams while
leaving the production model and its parameter names untouched:

* planned messages take ``RelationGatedMessage``'s eager formulation;
* cell/window/action/head stages run ``AxisMix -> FFN -> optional FiLM``;
* the two action relation tables use literal indexed lookups; and
* state/action latent passes reject their whole-pass fusion eligibility.

The fused and literal models are exact state-dict clones over real stack-939
prefixes.  In fp32 the only expected disagreement is floating-point
reassociation, so every forward tensor and every named parameter gradient is
gated at 2e-4 relative.  Mathematically zero gradients (the latent key biases)
are measured against the model-wide gradient scale, as the numerical register
requires, rather than divided by zero.  The bf16 arm is deliberately not a
fused-vs-reference gate: it reports one fused run against its fp32 anchor and
gates only finiteness and successful fused dispatch.
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
    action_encoder,
    class_embedding,
    fused_equivariant,
    fused_latent,
    fused_message,
    heads,
    messages,
    state_trunk,
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
    reason="whole-stage parity must exercise the registered CUDA operators",
)

ModelFactory = Callable[[], MantisACT]


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


def _literal_equivariant_stage(
    state,
    mix,
    ffn,
    *,
    film=None,
    phase_id=None,
):
    """Sections 12.4/13.2 as the three original module calls."""
    if (film is None) != (phase_id is None):
        raise ValueError("film and phase_id must be supplied together")
    result = ffn(mix(state))
    return result if film is None else film(result, phase_id)


def _literal_class_pair_embedding(
    post_weight: Tensor,
    status_weight: Tensor,
    post_index: Tensor,
    status_index: Tensor,
    _post_ptr: Tensor,
    _post_rows: Tensor,
    _status_ptr: Tensor,
    _status_rows: Tensor,
) -> Tensor:
    """Section 19.2's two table lookups, independent of the CSR backward."""
    return post_weight.index_select(0, post_index.long()) + status_weight.index_select(
        0, status_index.long()
    )


class _NeverPlanned:
    """A type no production edge object has, forcing the eager message branch."""


def _forbidden_class_launch(*_args, **_kwargs):
    raise AssertionError("the literal reference reached the fused class-pair kernel")


@contextmanager
def _literal_reference_path():
    """Restore every whole-stage pre-fusion composition for one scoped run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(messages, "PlannedEdges", _NeverPlanned)
        for module in (state_trunk, action_encoder, heads):
            patch.setattr(module, "run_equivariant_stage", _literal_equivariant_stage)
        patch.setattr(
            action_encoder, "class_pair_embedding", _literal_class_pair_embedding
        )
        patch.setattr(class_embedding, "_launch_forward", _forbidden_class_launch)
        patch.setattr(class_embedding, "_launch_backward", _forbidden_class_launch)

        # These predicates are LatentPass's production dispatch boundary;
        # refusing eligibility preserves its literal read/mix/broadcast body.
        for name in ("supports_state_pass", "supports_action_pass"):
            patch.setattr(fused_latent, name, lambda _module: False)
        yield


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


def _capture(
    factory: ModelFactory,
    batch: PackedACTBatch,
    *,
    bf16: bool = False,
) -> Snapshot:
    model = factory().to("cuda").eval()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        output = model(batch, MASS_FLOOR)
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


def _reset_whole_stage_proof() -> None:
    for module in (fused_message, fused_equivariant, fused_latent):
        module.clear_failure_caches()
        module.reset_launch_stats()
    class_embedding._FAILED_SHAPES.clear()
    class_embedding._FAILED_BACKWARD_SHAPES.clear()


def _assert_empty_whole_stage_failures() -> None:
    for name, module in (
        ("fused_message", fused_message),
        ("fused_equivariant", fused_equivariant),
        ("fused_latent", fused_latent),
    ):
        for cache_name in ("_FAILED_FORWARD_SHAPES", "_FAILED_BACKWARD_SHAPES"):
            cache = getattr(module, cache_name)
            assert not cache, {"family": name, "cache": cache_name, "failures": cache}
    assert not class_embedding._FAILED_SHAPES
    assert not class_embedding._FAILED_BACKWARD_SHAPES


def _assert_no_whole_stage_launches() -> None:
    for name, module in (
        ("fused_message", fused_message),
        ("fused_equivariant", fused_equivariant),
        ("fused_latent", fused_latent),
    ):
        stats = module.launch_stats()
        assert all(count == 0 for count in stats.values()), {name: stats}
    _assert_empty_whole_stage_failures()


def _assert_successful_whole_stage_launches() -> None:
    for name, module in (
        ("fused_message", fused_message),
        ("fused_equivariant", fused_equivariant),
    ):
        stats = module.launch_stats()
        for stage in ("forward", "backward"):
            eligible = stats[f"{stage}_eligible"]
            launched = stats[f"{stage}_launched"]
            assert eligible == launched and launched > 0, {name: stats}

    latent_stats = fused_latent.launch_stats()
    for variant in ("state", "action"):
        for stage in ("forward", "backward"):
            eligible = latent_stats[f"{variant}_{stage}_eligible"]
            launched = latent_stats[f"{variant}_{stage}_launched"]
            assert eligible == launched and launched > 0, {
                "fused_latent": latent_stats
            }
    _assert_empty_whole_stage_failures()


@contextmanager
def _count_class_embedding_launches():
    originals = {
        "_supported": class_embedding._supported,
        "_launch_forward": class_embedding._launch_forward,
        "_launch_backward": class_embedding._launch_backward,
    }
    counts = {"eligible": 0, "forward": 0, "backward": 0}

    def supported(*args, **kwargs):
        accepted = originals["_supported"](*args, **kwargs)
        counts["eligible"] += int(bool(accepted))
        return accepted

    def launch(kind: str):
        original = originals[f"_launch_{kind}"]

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            counts[kind] += 1
            return result

        return wrapped

    class_embedding._supported = supported
    class_embedding._launch_forward = launch("forward")
    class_embedding._launch_backward = launch("backward")
    try:
        yield counts
    finally:
        for name, original in originals.items():
            setattr(class_embedding, name, original)


def _reference_capture(factory: ModelFactory, batch: PackedACTBatch) -> Snapshot:
    _reset_whole_stage_proof()
    with _literal_reference_path():
        snapshot = _capture(factory, batch)
    _assert_no_whole_stage_launches()
    return snapshot


def _fused_capture(
    factory: ModelFactory, batch: PackedACTBatch, *, bf16: bool = False
) -> Snapshot:
    _reset_whole_stage_proof()
    with _count_class_embedding_launches() as class_counts:
        snapshot = _capture(factory, batch, bf16=bf16)
    _assert_successful_whole_stage_launches()
    assert class_counts["forward"] > 0 and class_counts["backward"] > 0, class_counts
    assert class_counts["eligible"] == (
        class_counts["forward"] + class_counts["backward"]
    ), class_counts
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


def _run_parity(
    factory: ModelFactory, cfg: MantisACTConfig, ply: int, batch_size: int
) -> None:
    batch = _real_batch(cfg, ply, batch_size).to("cuda")
    expected = _reference_capture(factory, batch)
    actual = _fused_capture(factory, batch)
    _assert_fp32_parity(actual, expected)
    del batch, expected, actual
    torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.parametrize("ply", PLIES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_random_weight_fp32_whole_model_matches_literal_stages(
    random_state: Mapping[str, Tensor], ply: int, batch_size: int
) -> None:
    """Only fp32 reassociation may separate the two exact state-dict clones."""
    _run_parity(_factory_from_state(FULL, random_state), FULL, ply, batch_size)


@requires_cuda
def test_real_checkpoint_fp32_whole_model_matches_literal_stages() -> None:
    """Run the §36 checkpoint oracle when its owner supplies the artifact."""
    raw_path = os.environ.get("MANTIS_ACT_CHECKPOINT", "").strip()
    if not raw_path:
        pytest.skip(
            "MANTIS_ACT_CHECKPOINT is unset; set it to a full_act_v4 checkpoint "
            "to run the required real-checkpoint whole-stage oracle"
        )
    path = Path(raw_path)
    assert path.is_file(), f"MANTIS_ACT_CHECKPOINT does not name a file: {path}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    probe = MantisACT.from_checkpoint(payload).eval()
    assert probe.cfg == FULL, (
        "the whole-stage checkpoint oracle requires full_act_v4, got "
        f"{probe.cfg}"
    )

    def factory() -> MantisACT:
        return MantisACT.from_checkpoint(payload).eval()

    del probe
    _run_parity(factory, FULL, ply=161, batch_size=8)


@requires_cuda
def test_bf16_error_is_reported_against_fp32_anchor_not_itself(
    random_state: Mapping[str, Tensor], record_property
) -> None:
    """Document bf16 drift; only finiteness and real fused launches are gates."""
    factory = _factory_from_state(FULL, random_state)
    batch = _real_batch(FULL, ply=161, batch_size=8).to("cuda")
    anchor = _fused_capture(factory, batch)
    bf16 = _fused_capture(factory, batch, bf16=True)
    output_metrics, gradient_metrics = _parity_metrics(bf16, anchor)
    worst_output = max(output_metrics.items(), key=lambda item: item[1])
    worst_gradient = max(gradient_metrics.items(), key=lambda item: item[1])
    record_property("bf16_fp32_worst_output", worst_output[1])
    record_property("bf16_fp32_worst_output_name", worst_output[0])
    record_property("bf16_fp32_worst_gradient", worst_gradient[1])
    record_property("bf16_fp32_worst_gradient_name", worst_gradient[0])
    del batch, anchor, bf16
    torch.cuda.empty_cache()
