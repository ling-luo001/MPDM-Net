import torch
import yaml

from models.generator import MambaSEUNet


def main():
    with open('recipes/Mamba-SEUNet/Mamba-SEUNet.yaml', encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaSEUNet(cfg).to(device).eval()
    batch_size = 2
    frames = 32
    freq_bins = cfg['stft_cfg']['n_fft'] // 2 + 1
    x_mag = torch.rand(batch_size, freq_bins, frames, device=device)
    x_pha = (2.0 * torch.pi * torch.rand_like(x_mag)) - torch.pi

    with torch.no_grad():
        denoised_mag, pred_phase, denoised_complex = model(x_mag, x_pha)

    expected_spectrum_shape = (batch_size, freq_bins, frames)
    assert denoised_mag.shape == expected_spectrum_shape
    assert pred_phase.shape == expected_spectrum_shape
    assert denoised_complex.shape == (*expected_spectrum_shape, 2)
    assert torch.isfinite(denoised_mag).all()
    assert torch.isfinite(pred_phase).all()
    assert torch.isfinite(denoised_complex).all()

    aux = model.latest_aux
    assert aux['restoration_gate'].shape == (batch_size, 1, frames, freq_bins)
    assert torch.allclose(denoised_complex, aux['coarse_complex'], atol=1e-6)
    assert torch.allclose(
        denoised_mag,
        torch.linalg.vector_norm(denoised_complex, dim=-1),
        atol=1e-6
    )
    expected_gate = torch.sigmoid(
        torch.tensor(model.complex_residual_gate_bias, device=device)
    )
    assert torch.allclose(aux['restoration_gate'], expected_gate.expand_as(aux['restoration_gate']))
    assert aux['suppression_context_scale'].item() == 0.0

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'device={device}')
    print(f'parameters={parameter_count:,}')
    print(
        f'magnitude={tuple(denoised_mag.shape)}, '
        f'phase={tuple(pred_phase.shape)}, complex={tuple(denoised_complex.shape)}'
    )
    print('progressive suppression-restoration smoke test passed')


if __name__ == '__main__':
    main()
