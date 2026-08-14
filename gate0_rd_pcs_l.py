"""Local Gate 0 for RD-PCS-L. This script never starts training or loads weights."""

from __future__ import annotations

import ast
import math
import sys
import time
import types
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parent
RECIPE_PATH = ROOT / "recipes" / "RD-PCS-L" / "RD-PCS-L.yaml"
BASELINE_RECIPE_PATH = ROOT / "recipes" / "Mamba-SEUNet" / "Mamba-SEUNet.yaml"
EXPECTED_EXPERIMENT_NAME = "rd_pcs_l_h24_t3_m3_pcs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    require(isinstance(cfg, dict), f"Expected a YAML mapping: {path}")
    return cfg


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def check_config() -> tuple[dict, dict]:
    candidate = load_yaml(RECIPE_PATH)
    baseline = load_yaml(BASELINE_RECIPE_PATH)
    expected = {
        "hid_feature": 24,
        "num_tfmamba": 3,
        "num_mid_pairs": 3,
        "restoration_width_ratio": 1.0,
    }
    for key, value in expected.items():
        require(candidate["model_cfg"].get(key) == value, f"Unexpected {key}")
    require(candidate["training_cfg"].get("use_PCS400") is True, "PCS must be enabled")
    experiment = candidate.get("experiment_cfg", {})
    require(experiment.get("name") == EXPECTED_EXPERIMENT_NAME, "Wrong experiment name")
    require(
        experiment.get("output_dir") == f"exp/{EXPECTED_EXPERIMENT_NAME}",
        "RD-PCS-L needs an independent output directory",
    )
    require(
        experiment.get("log_dir") == f"exp/{EXPECTED_EXPERIMENT_NAME}/logs",
        "RD-PCS-L needs an independent log directory",
    )
    normalized = deepcopy(candidate)
    normalized.pop("experiment_cfg")
    normalized["env_setting"]["dist_cfg"]["dist_url"] = baseline["env_setting"]["dist_cfg"]["dist_url"]
    normalized["training_cfg"]["use_PCS400"] = baseline["training_cfg"]["use_PCS400"]
    normalized["model_cfg"]["hid_feature"] = baseline["model_cfg"]["hid_feature"]
    normalized["model_cfg"]["num_tfmamba"] = baseline["model_cfg"]["num_tfmamba"]
    require(
        normalized == baseline,
        "Candidate recipe drifted beyond capacity, PCS, and experiment isolation",
    )
    forbidden_reuse_keys = {
        "resume",
        "resume_from",
        "resume_step",
        "pretrained",
        "pretrained_path",
        "weights",
        "weight_path",
        "checkpoint_path",
    }
    found = forbidden_reuse_keys.intersection(walk_keys(candidate))
    require(not found, f"Weight reuse keys are forbidden in the recipe: {sorted(found)}")
    print(f"[PASS] config: {RECIPE_PATH.relative_to(ROOT)}")
    print("       controlled diff: capacity + PCS + isolated experiment namespace/port")
    print(f"       future output/log namespace: exp/{EXPECTED_EXPERIMENT_NAME}")
    print("       future mini mode: add --mini to the documented launch command")
    return candidate, baseline


def check_no_weight_loading_calls() -> None:
    forbidden_calls = {"load_ckpts", "load_state_dict", "load_state_dict_hf", "torch.load"}
    violations = []
    for path in (Path(__file__), ROOT / "models" / "pcs400.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                call_name = f"{node.func.value.id}.{node.func.attr}"
            else:
                continue
            if call_name in forbidden_calls:
                violations.append(f"{path.name}:{node.lineno}:{call_name}")
    require(not violations, f"Weight-loading calls are forbidden in Gate 0 code: {violations}")
    print("[PASS] Gate 0 code contains no checkpoint/weight loading calls")


def check_pcs() -> None:
    from models.pcs400 import cal_pcs

    sample_count = 1600
    time = np.arange(sample_count, dtype=np.float32) / np.float32(16000.0)
    signals = {
        "silence": np.zeros(sample_count, dtype=np.float32),
        "tiny": (np.float32(1e-30) * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32),
        "ordinary": (
            0.35 * np.sin(2.0 * np.pi * 220.0 * time)
            + 0.12 * np.sin(2.0 * np.pi * 730.0 * time)
        ).astype(np.float32),
    }
    outputs = {name: cal_pcs(signal) for name, signal in signals.items()}
    for name, output in outputs.items():
        require(output.shape == signals[name].shape, f"PCS length changed for {name}")
        require(output.dtype == signals[name].dtype, f"PCS dtype changed for {name}")
        require(np.isfinite(output).all(), f"PCS produced NaN/Inf for {name}")
    require(np.count_nonzero(outputs["silence"]) == 0, "Silence must remain silent")
    require(np.max(np.abs(outputs["ordinary"])) > 0.99, "Ordinary PCS output is trivial")
    require(np.std(outputs["ordinary"]) > 1e-2, "Ordinary PCS output lacks variation")
    normalized_input = signals["ordinary"] / np.max(np.abs(signals["ordinary"]))
    require(
        not np.allclose(outputs["ordinary"], normalized_input, rtol=1e-3, atol=1e-4),
        "PCS did not produce a non-trivial spectral shaping",
    )
    print("[PASS] PCS: silence/tiny/ordinary are finite; length and float32 dtype preserved")


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear) and module.bias is not None:
        if not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, parameter in module.named_parameters():
            if name in {"out_proj.weight", "fc2.weight"}:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                with torch.no_grad():
                    parameter /= math.sqrt(n_residuals_per_layer * n_layer)


def install_structure_stub_if_needed() -> bool:
    """Install a shape-preserving Mamba stub only when its CUDA extension is absent."""
    try:
        __import__("selective_scan_cuda")
        return False
    except ImportError:
        pass

    package = types.ModuleType("mamba_ssm")
    package.__path__ = [str(ROOT / "mamba_ssm")]
    sys.modules["mamba_ssm"] = package
    sys.modules["selective_scan_cuda"] = types.ModuleType("selective_scan_cuda")
    if not hasattr(nn, "RMSNorm"):
        nn.RMSNorm = nn.LayerNorm

    layernorm_module = types.ModuleType("mamba_ssm.ops.triton.layernorm")
    layernorm_module.RMSNorm = nn.LayerNorm
    layernorm_module.layer_norm_fn = None
    layernorm_module.rms_norm_fn = None
    sys.modules["mamba_ssm.ops.triton.layernorm"] = layernorm_module

    mixer_module = types.ModuleType("mamba_ssm.models.mixer_seq_simple")
    mixer_module._init_weights = _init_weights
    sys.modules["mamba_ssm.models.mixer_seq_simple"] = mixer_module

    from mamba_ssm.modules.mamba_simple import Mamba

    def shape_only_forward(self, hidden_states, inference_params=None):
        del inference_params
        zero = sum((parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()), 0.0)
        return hidden_states + zero

    Mamba.forward = shape_only_forward
    return True


class CapturedDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pcs = kwargs["pcs"]


def check_dataset_routing(cfg: dict) -> None:
    import train

    train.VCTKDemandDataset = CapturedDataset
    train.Val_Dataset = CapturedDataset
    training_set = train.create_dataset(cfg, train=True, split=True, device="cpu")
    validation_set = train.create_val_dataset(cfg, train=False, split=False, device="cpu")
    require(training_set.pcs is True, "Training dataset did not receive PCS=True")
    require(validation_set.pcs is False, "Validation dataset must receive PCS=False")
    print("[PASS] dataset routing: training PCS=True; validation PCS=False")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def construct_and_count(model_class, cfg: dict, label: str) -> tuple[nn.Module, int, float]:
    started = time.perf_counter()
    model = model_class(cfg)
    elapsed = time.perf_counter() - started
    parameter_count = count_parameters(model)
    print(f"       {label}: {parameter_count:,} parameters; construct={elapsed:.3f}s")
    return model, parameter_count, elapsed


def check_models(
    candidate_cfg: dict, baseline_cfg: dict, stub_active: bool
) -> tuple[int, int, int, float, float]:
    from models.generator import MambaSEUNet

    intermediate_cfg = deepcopy(candidate_cfg)
    intermediate_cfg["model_cfg"]["hid_feature"] = 20

    print("[INFO] capacity construction:")
    torch.manual_seed(1234)
    baseline, baseline_params, _ = construct_and_count(MambaSEUNet, baseline_cfg, "H16/N2 baseline")
    del baseline

    torch.manual_seed(1234)
    intermediate, intermediate_params, _ = construct_and_count(
        MambaSEUNet, intermediate_cfg, "H20/N3 intermediate"
    )
    del intermediate

    torch.manual_seed(1234)
    candidate, candidate_params, _ = construct_and_count(MambaSEUNet, candidate_cfg, "H24/N3 candidate")
    require(
        baseline_params < intermediate_params < candidate_params,
        "Expected H16/N2 < H20/N3 < H24/N3 parameter counts",
    )
    require(4_000_000 <= candidate_params <= 7_000_000, "Candidate must stay in the 4-7M target band")
    print("[PASS] capacity order: H16/N2 < H20/N3 < H24/N3")

    candidate.train()
    noisy_mag = torch.rand(1, 256, 4)
    noisy_pha = (torch.rand(1, 256, 4) - 0.5) * (2.0 * math.pi)
    forward_started = time.perf_counter()
    outputs = candidate(noisy_mag, noisy_pha)
    forward_elapsed = time.perf_counter() - forward_started
    expected_shapes = ((1, 256, 4), (1, 256, 4), (1, 256, 4, 2))
    require(tuple(output.shape for output in outputs) == expected_shapes, "Unexpected output shapes")
    require(all(torch.isfinite(output).all() for output in outputs), "Forward produced NaN/Inf")
    loss = sum(output.square().mean() for output in outputs)
    backward_started = time.perf_counter()
    loss.backward()
    backward_elapsed = time.perf_counter() - backward_started
    gradients = [parameter.grad for parameter in candidate.parameters() if parameter.grad is not None]
    require(gradients, "Backward produced no gradients")
    require(all(torch.isfinite(gradient).all() for gradient in gradients), "Backward produced NaN/Inf")
    require(any(torch.count_nonzero(gradient).item() for gradient in gradients), "All gradients are zero")
    validation_kind = "STRUCTURE-ONLY STUB" if stub_active else "RUNTIME CUDA MAMBA"
    print(
        f"[PASS] small forward/backward: shapes={expected_shapes}; finite ({validation_kind}); "
        f"forward={forward_elapsed:.3f}s; backward={backward_elapsed:.3f}s"
    )
    return baseline_params, intermediate_params, candidate_params, forward_elapsed, backward_elapsed


def main() -> None:
    sys.path.insert(0, str(ROOT))
    candidate_cfg, baseline_cfg = check_config()
    check_no_weight_loading_calls()
    check_pcs()
    stub_active = install_structure_stub_if_needed()
    if stub_active:
        print("[NOTICE] selective_scan_cuda unavailable: explicit shape-only Mamba test stub active")
        print("         This is structural validation, not a real Mamba runtime validation.")
    check_dataset_routing(candidate_cfg)
    baseline_params, intermediate_params, candidate_params, forward_elapsed, backward_elapsed = check_models(
        candidate_cfg, baseline_cfg, stub_active
    )
    print("[PASS] no checkpoint/weight reuse configuration is used by Gate 0")
    print(
        "GATE 0 PASS | "
        f"H16/N2={baseline_params:,} | H20/N3={intermediate_params:,} | "
        f"H24/N3={candidate_params:,} | forward={forward_elapsed:.3f}s | "
        f"backward={backward_elapsed:.3f}s | "
        f"mamba={'stub-structure-only' if stub_active else 'runtime'}"
    )


if __name__ == "__main__":
    main()
