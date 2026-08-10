"""Run the PGRT-MPDM Gate 0 checks on a CUDA host."""

import argparse
import copy
import gc
import json
import statistics
import time

import torch

from models.discriminator import MetricDiscriminator
from models.generator import MambaSEUNet
from utils.util import load_config


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def assert_state_equal(reference, candidate, ignored_prefix=None):
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    compared = 0
    for name, tensor in reference_state.items():
        if ignored_prefix is not None and name.startswith(ignored_prefix):
            continue
        if name not in candidate_state:
            raise AssertionError("missing paired state: " + name)
        if not torch.equal(tensor, candidate_state[name]):
            raise AssertionError("unpaired initialization: " + name)
        compared += 1
    return compared


def build_paired_models(config, seed):
    baseline_config = copy.deepcopy(config)
    baseline_config["model_cfg"]["pgrt_enabled"] = False
    pgrt_config = copy.deepcopy(config)
    pgrt_config["model_cfg"]["pgrt_enabled"] = True

    seed_everything(seed)
    baseline = MambaSEUNet(baseline_config)
    baseline_discriminator = MetricDiscriminator()

    seed_everything(seed)
    pgrt = MambaSEUNet(pgrt_config)
    pgrt_discriminator = MetricDiscriminator()

    common_states = assert_state_equal(baseline, pgrt)
    discriminator_states = assert_state_equal(
        baseline_discriminator, pgrt_discriminator
    )
    return baseline, pgrt, common_states, discriminator_states


def make_inputs(batch_size, frequency_bins, frames, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    magnitude = 0.05 + torch.rand(
        batch_size, frequency_bins, frames, generator=generator
    )
    phase = (2.0 * torch.rand(
        batch_size, frequency_bins, frames, generator=generator
    ) - 1.0) * torch.pi
    return magnitude, phase


def release_cuda(model):
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def run_outputs(model, magnitude_cpu, phase_cpu, device):
    model.to(device).eval()
    magnitude = magnitude_cpu.to(device)
    phase = phase_cpu.to(device)
    with torch.no_grad():
        outputs = tuple(tensor.cpu() for tensor in model(magnitude, phase))
    release_cuda(model)
    return outputs


def run_input_vjp(model, magnitude_cpu, phase_cpu, device):
    model.to(device).eval()
    magnitude = magnitude_cpu.to(device).requires_grad_(True)
    phase = phase_cpu.to(device).requires_grad_(True)
    outputs = model(magnitude, phase)
    objective = (
        outputs[0].mean()
        + 0.37 * outputs[1].mean()
        + 0.13 * outputs[2].mean()
    )
    gradients = torch.autograd.grad(objective, (magnitude, phase))
    gradients = tuple(gradient.detach().cpu() for gradient in gradients)
    release_cuda(model)
    return gradients


def assert_exact_pairs(reference, candidate, label):
    if len(reference) != len(candidate):
        raise AssertionError(label + " tuple length mismatch")
    maxima = []
    for reference_tensor, candidate_tensor in zip(reference, candidate):
        maxima.append((reference_tensor - candidate_tensor).abs().max().item())
        if not torch.equal(reference_tensor, candidate_tensor):
            raise AssertionError(label + " is not bitwise identical")
    return maxima


def training_iteration(model, magnitude, phase):
    model.zero_grad(set_to_none=True)
    outputs = model(magnitude, phase)
    loss = (
        outputs[0].square().mean()
        + outputs[1].square().mean()
        + outputs[2].square().mean()
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite profiling loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite gradient: " + name)


def profile_training(model, magnitude_cpu, phase_cpu, device, iterations):
    model.to(device).train()
    magnitude = magnitude_cpu.to(device)
    phase = phase_cpu.to(device)
    training_iteration(model, magnitude, phase)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        training_iteration(model, magnitude, phase)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    release_cuda(model)
    return {
        "median_seconds": statistics.median(elapsed),
        "samples_seconds": elapsed,
        "peak_bytes": peak_bytes,
        "peak_mib": peak_bytes / (1024.0 ** 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="recipes/Mamba-SEUNet/PGRT-MPDM-v1-mini.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--identity-frames", type=int, default=64)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--max-time-ratio", type=float, default=1.25)
    parser.add_argument("--max-memory-ratio", type=float, default=1.25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Gate 0 runtime checks require CUDA")
    device = torch.device(args.device)
    config = load_config(args.config)
    seed = int(config["env_setting"]["seed"])
    frequency_bins = int(config["stft_cfg"]["n_fft"]) // 2 + 1
    profile_frames = int(config["training_cfg"]["segment_size"]) // int(
        config["stft_cfg"]["hop_size"]
    ) + 1

    baseline, pgrt, common_states, discriminator_states = build_paired_models(
        config, seed
    )
    baseline_parameters = count_parameters(baseline)
    pgrt_parameters = count_parameters(pgrt)
    added_parameters = pgrt_parameters - baseline_parameters
    if added_parameters > 120_000:
        raise AssertionError("PGRT exceeds the 120k parameter gate")

    identity_inputs = make_inputs(
        1, frequency_bins, args.identity_frames, seed + 1
    )
    baseline_outputs = run_outputs(baseline, *identity_inputs, device)
    pgrt_outputs = run_outputs(pgrt, *identity_inputs, device)
    output_maxima = assert_exact_pairs(
        baseline_outputs, pgrt_outputs, "zero-injection output"
    )

    baseline_vjp = run_input_vjp(baseline, *identity_inputs, device)
    pgrt_vjp = run_input_vjp(pgrt, *identity_inputs, device)
    vjp_maxima = assert_exact_pairs(
        baseline_vjp, pgrt_vjp, "zero-injection input VJP"
    )

    with torch.no_grad():
        pgrt.pgrt.stage_branch_scales.fill_(0.1)
    profile_inputs = make_inputs(
        int(config["training_cfg"]["batch_size"]),
        frequency_bins,
        profile_frames,
        seed + 2,
    )
    baseline_profile = profile_training(
        baseline, *profile_inputs, device, args.profile_iterations
    )
    pgrt_profile = profile_training(
        pgrt, *profile_inputs, device, args.profile_iterations
    )
    time_ratio = (
        pgrt_profile["median_seconds"] / baseline_profile["median_seconds"]
    )
    memory_ratio = pgrt_profile["peak_bytes"] / baseline_profile["peak_bytes"]

    report = {
        "baseline_parameters": baseline_parameters,
        "pgrt_parameters": pgrt_parameters,
        "added_parameters": added_parameters,
        "paired_generator_state_tensors": common_states,
        "paired_discriminator_state_tensors": discriminator_states,
        "zero_output_max_abs": output_maxima,
        "zero_vjp_max_abs": vjp_maxima,
        "profile_batch_size": int(config["training_cfg"]["batch_size"]),
        "profile_frequency_bins": frequency_bins,
        "profile_frames": profile_frames,
        "baseline_profile": baseline_profile,
        "pgrt_profile": pgrt_profile,
        "time_ratio": time_ratio,
        "memory_ratio": memory_ratio,
    }
    print(json.dumps(report, indent=2))

    if time_ratio > args.max_time_ratio:
        raise AssertionError("PGRT exceeds the training-time ratio gate")
    if memory_ratio > args.max_memory_ratio:
        raise AssertionError("PGRT exceeds the peak-memory ratio gate")
    print("PASS PGRT Gate 0")


if __name__ == "__main__":
    main()
