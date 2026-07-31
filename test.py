import torch
import torch.nn.functional as F
import yaml

from models.generator import MambaSEUNet, PromptNAFSEUNet


def main():
    with open('recipes/Mamba-SEUNet/Mamba-SEUNet.yaml', encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaSEUNet(cfg).to(device)
    assert isinstance(model, PromptNAFSEUNet)

    batch_size = 2
    frames = 31  # Deliberately exercise internal padding and output cropping.
    freq_bins = cfg['stft_cfg']['n_fft'] // 2 + 1
    noisy_mag = torch.rand(batch_size, freq_bins, frames, device=device)
    noisy_pha = (2.0 * torch.pi * torch.rand_like(noisy_mag)) - torch.pi

    denoised_mag, pred_phase, denoised_complex = model(noisy_mag, noisy_pha)
    expected_spectrum_shape = (batch_size, freq_bins, frames)
    assert denoised_mag.shape == expected_spectrum_shape
    assert pred_phase.shape == expected_spectrum_shape
    assert denoised_complex.shape == (*expected_spectrum_shape, 2)
    assert torch.isfinite(denoised_mag).all()
    assert torch.isfinite(pred_phase).all()
    assert torch.isfinite(denoised_complex).all()

    noisy_complex = torch.stack(
        [noisy_mag * torch.cos(noisy_pha), noisy_mag * torch.sin(noisy_pha)],
        dim=-1,
    )
    assert torch.allclose(denoised_complex, noisy_complex, atol=1e-6)
    assert torch.allclose(
        denoised_mag,
        torch.linalg.vector_norm(denoised_complex, dim=-1),
        atol=1e-6,
    )

    aux = model.latest_aux
    prompt_count = cfg['model_cfg']['prompt_count']
    assert aux['complex_residual'].shape == (*expected_spectrum_shape, 2)
    assert aux['global_prompt_weights'].shape == (batch_size, prompt_count)
    assert aux['temporal_prompt_weights'].shape == (batch_size, prompt_count, frames)
    assert aux['spectral_prompt_weights'].shape == (batch_size, prompt_count, freq_bins)
    assert aux['temporal_noise_log_ratio'].shape == (batch_size, frames)
    assert aux['spectral_noise_log_ratio'].shape == (batch_size, freq_bins)
    assert torch.allclose(
        aux['global_prompt_weights'].sum(dim=1),
        torch.ones(batch_size, device=device),
        atol=1e-5,
    )
    assert torch.allclose(
        aux['temporal_prompt_weights'].sum(dim=1),
        torch.ones(batch_size, frames, device=device),
        atol=1e-5,
    )
    assert torch.allclose(
        aux['spectral_prompt_weights'].sum(dim=1),
        torch.ones(batch_size, freq_bins, device=device),
        atol=1e-5,
    )

    clean_proxy = noisy_complex * 0.8
    noise_energy = (noisy_complex - clean_proxy).square().sum(dim=-1)
    mixture_energy = noisy_mag.square()
    temporal_target = (
        torch.log(noise_energy.mean(dim=1) + 1e-6)
        - torch.log(mixture_energy.mean(dim=1) + 1e-6)
    ).clamp(-6.0, 3.0)
    spectral_target = (
        torch.log(noise_energy.mean(dim=2) + 1e-6)
        - torch.log(mixture_energy.mean(dim=2) + 1e-6)
    ).clamp(-6.0, 3.0)
    smoke_loss = (
        denoised_mag.mean()
        + F.smooth_l1_loss(aux['temporal_noise_log_ratio'], temporal_target)
        + F.smooth_l1_loss(aux['spectral_noise_log_ratio'], spectral_target)
    )
    smoke_loss.backward()
    assert torch.isfinite(model.complex_residual_head.weight.grad).all()
    assert torch.isfinite(model.prompt_estimator.temporal_noise_profile.weight.grad).all()
    assert torch.isfinite(model.prompt_estimator.spectral_noise_profile.weight.grad).all()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'device={device}')
    print(f'parameters={parameter_count:,}')
    print(
        f'magnitude={tuple(denoised_mag.shape)}, '
        f'phase={tuple(pred_phase.shape)}, complex={tuple(denoised_complex.shape)}'
    )
    print('tri-granular prompt NAF restoration smoke test passed')


if __name__ == '__main__':
    main()
