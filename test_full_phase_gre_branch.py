import math

import pytest
import torch

from models.generator import ComplexPatchEmbed2D, ComplexPhaseDecoderHead, MambaSEUNet
from models.mamba_block import (
    ComplexConcatFuse2D,
    ComplexConv2dNoBias,
    ComplexDownsample2D,
    ComplexEqTMambaBlock,
    ComplexRMSNorm2D,
    ComplexUpsample2D,
)


def _get_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for full phase GRE tests.")
    return torch.device("cuda")


def _get_cfg():
    return {
        "model_cfg": {
            "hid_feature": 16,
            "dense_channel": 16,
            "compress_factor": 0.3,
            "num_tfmamba": 2,
            "num_mid_pairs": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 4,
            "norm_epsilon": 1e-5,
            "beta": 2.0,
            "input_channel": 2,
            "output_channel": 1,
            "cross_pool_t": 1,
            "cross_pool_f": 1,
            "cross_sparse_window": 64,
            "cross_global_window": 8,
            "cross_sparsity": 0.9,
            "eq_mamba_res_scale": 1.0,
            "eq_mamba_dropout": 0.0,
            "eq_mamba_bidirectional": True,
            "eq_mamba_use_complex_conv": True,
            "full_phase_stem_freq_stride": 2,
            "full_phase_stem_use_post_conv": False,
            "phase_head_upsample_mode": "bilinear",
            "gre_fusion_scale": 1.0,
            "use_complex_phase_refine_ffn": True,
            "pha_ffn_dropout": 0.0,
            "pha_ffn_res_scale": 1.0,
            "pha_ffn_expansion": 4,
            "pha_ffn_use_complex_conv": True,
            "pha_ffn_kernel_size": 3,
        },
        "training_cfg": {
            "segment_size": 32000,
        },
        "stft_cfg": {
            "n_fft": 510,
        },
    }


def test_complex_2d_modules_shape_backward():
    device = _get_device()
    b, c, t, f = 2, 8, 16, 12
    x_r = torch.randn(b, c, t, f, device=device, requires_grad=True)
    x_i = torch.randn(b, c, t, f, device=device, requires_grad=True)

    conv = ComplexConv2dNoBias(c, c, kernel_size=3, padding=1).to(device)
    norm = ComplexRMSNorm2D(c).to(device)
    patch = ComplexPatchEmbed2D(c).to(device)
    down = ComplexDownsample2D(c, c * 2).to(device)
    up = ComplexUpsample2D(c * 2, c).to(device)
    fuse = ComplexConcatFuse2D(c * 2, c).to(device)
    head = ComplexPhaseDecoderHead(c).to(device)

    y_r, y_i = conv(x_r, x_i)
    y_r, y_i = norm(y_r, y_i)
    p_r, p_i = patch(y_r, y_i)
    d_r, d_i = down(p_r, p_i)
    u_r, u_i = up(d_r, d_i)
    f_r, f_i = fuse(u_r, u_i, y_r, y_i)
    pred_cos, pred_sin = head(f_r, f_i, target_size=(t, f))

    assert y_r.shape == x_r.shape
    assert y_i.shape == x_i.shape
    assert p_r.shape == x_r.shape
    assert p_i.shape == x_i.shape
    assert d_r.shape == (b, c * 2, t // 2, f // 2)
    assert d_i.shape == (b, c * 2, t // 2, f // 2)
    assert u_r.shape == x_r.shape
    assert u_i.shape == x_i.shape
    assert f_r.shape == x_r.shape
    assert f_i.shape == x_i.shape
    assert pred_cos.shape == (b, t, f)
    assert pred_sin.shape == (b, t, f)
    assert torch.isfinite(pred_cos).all()
    assert torch.isfinite(pred_sin).all()

    loss = pred_cos.abs().mean() + pred_sin.abs().mean()
    loss.backward()


def test_complex_phase_head_equivariance():
    device = _get_device()
    b, c, t, f = 2, 12, 20, 17
    x_r = torch.randn(b, c, t, f, device=device)
    x_i = torch.randn(b, c, t, f, device=device)

    head = ComplexPhaseDecoderHead(c).to(device)
    head.eval()

    theta = torch.rand(1, device=device) * 2 * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x2_r = cos_t * x_r - sin_t * x_i
    x2_i = sin_t * x_r + cos_t * x_i

    target_size = (t, f)
    y1_c, y1_s = head(x_r, x_i, target_size=target_size)
    y2_c, y2_s = head(x2_r, x2_i, target_size=target_size)

    target_c = cos_t * y1_c - sin_t * y1_s
    target_s = sin_t * y1_c + cos_t * y1_s

    err = (
        (y2_c - target_c).abs().mean()
        + (y2_s - target_s).abs().mean()
    ) / (
        target_c.abs().mean()
        + target_s.abs().mean()
        + 1e-8
    )

    print(f"ComplexPhaseDecoderHead equivariance error: {err.item()}")
    assert err < 1e-3


def test_full_phase_branch_construction():
    device = _get_device()
    cfg = _get_cfg()
    model = MambaSEUNet(cfg).to(device)

    assert hasattr(model, "pha_complex_stem")
    assert hasattr(model, "gre_final_fusion")
    assert hasattr(model, "complex_phase_head")
    assert isinstance(model.pha_complex_TSMamba1_encoder[0], ComplexEqTMambaBlock)
    assert isinstance(model.pha_complex_TSMamba2_encoder[0], ComplexEqTMambaBlock)
    assert isinstance(model.pha_complex_refinement[0], ComplexEqTMambaBlock)


def test_no_middle_fusion_modules():
    device = _get_device()
    cfg = _get_cfg()
    model = MambaSEUNet(cfg).to(device)

    assert not hasattr(model, "mid_fusions")
    assert not hasattr(model, "mid_in_proj_mag")
    assert not hasattr(model, "mid_in_proj_pha")
    assert not hasattr(model, "mid_fusion_proj_mag")
    assert not hasattr(model, "mid_fusion_proj_pha")
    assert hasattr(model, "gre_final_fusion")
    assert hasattr(model, "complex_phase_head")


def test_full_model_forward_smoke():
    device = _get_device()
    cfg = _get_cfg()
    model = MambaSEUNet(cfg).to(device)
    model.eval()

    b = 1
    f = cfg["stft_cfg"]["n_fft"] // 2 + 1
    t = 64

    noisy_mag = torch.rand(b, f, t, device=device)
    noisy_pha = torch.randn(b, f, t, device=device)

    with torch.no_grad():
        denoised_mag, pred_pha, denoised_com = model(noisy_mag, noisy_pha)

    assert denoised_mag.shape == (b, f, t)
    assert pred_pha.shape == (b, f, t)
    assert denoised_com.shape == (b, f, t, 2)
    assert torch.isfinite(denoised_mag).all()
    assert torch.isfinite(pred_pha).all()
    assert torch.isfinite(denoised_com).all()


def test_phase_head_upsample_norm():
    device = _get_device()
    b, c, t, f = 2, 8, 20, 64
    x_r = torch.randn(b, c, t, f // 2, device=device)
    x_i = torch.randn(b, c, t, f // 2, device=device)

    head = ComplexPhaseDecoderHead(c, upsample_mode="bilinear").to(device)
    pred_cos, pred_sin = head(x_r, x_i, target_size=(t, f))

    assert pred_cos.shape == (b, t, f)
    assert pred_sin.shape == (b, t, f)
    assert torch.isfinite(pred_cos).all()
    assert torch.isfinite(pred_sin).all()

    norm = torch.sqrt(pred_cos ** 2 + pred_sin ** 2 + 1e-8)
    err = (norm - 1.0).abs().mean()
    assert err < 1e-3
