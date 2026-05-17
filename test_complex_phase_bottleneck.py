import math

import pytest
import torch

from models.cross import GREVSSMidFusion
from models.mamba_block import (
    ComplexEqTMambaBlock,
    ComplexEqFMambaBlock,
    PhaseMidToComplex2D,
    PhaseMidToReal2D,
)


def _get_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for complex phase bottleneck tests.")
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
            "gre_fusion_scale": 1.0,
            "pha_ffn_dropout": 0.0,
            "pha_ffn_res_scale": 1.0,
        },
        "training_cfg": {
            "segment_size": 32000,
        },
        "stft_cfg": {
            "n_fft": 510,
        },
    }


def test_complex_eq_mamba_blocks_shape_backward():
    device = _get_device()
    cfg = _get_cfg()
    c = 24
    x_r = torch.randn(2, c, 50, 33, device=device)
    x_i = torch.randn(2, c, 50, 33, device=device)
    tblock = ComplexEqTMambaBlock(cfg, c).to(device)
    fblock = ComplexEqFMambaBlock(cfg, c).to(device)

    y_r, y_i = tblock(x_r, x_i)
    z_r, z_i = fblock(x_r, x_i)

    assert y_r.shape == x_r.shape
    assert y_i.shape == x_i.shape
    assert z_r.shape == x_r.shape
    assert z_i.shape == x_i.shape
    assert torch.isfinite(y_r).all()
    assert torch.isfinite(y_i).all()
    assert torch.isfinite(z_r).all()
    assert torch.isfinite(z_i).all()

    loss = y_r.abs().mean() + y_i.abs().mean() + z_r.abs().mean() + z_i.abs().mean()
    loss.backward()


def _equivariance_error(block, x_r, x_i):
    theta = torch.rand(1, device=x_r.device) * 2 * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x2_r = cos_t * x_r - sin_t * x_i
    x2_i = sin_t * x_r + cos_t * x_i

    y1_r, y1_i = block(x_r, x_i)
    y2_r, y2_i = block(x2_r, x2_i)

    target_r = cos_t * y1_r - sin_t * y1_i
    target_i = sin_t * y1_r + cos_t * y1_i

    err = (
        (y2_r - target_r).abs().mean()
        + (y2_i - target_i).abs().mean()
    ) / (
        target_r.abs().mean()
        + target_i.abs().mean()
        + 1e-8
    )
    return err


def test_complex_eq_mamba_blocks_equivariance():
    device = _get_device()
    cfg = _get_cfg()
    c = 24
    x_r = torch.randn(2, c, 50, 33, device=device)
    x_i = torch.randn(2, c, 50, 33, device=device)

    tblock = ComplexEqTMambaBlock(cfg, c).to(device)
    fblock = ComplexEqFMambaBlock(cfg, c).to(device)
    tblock.eval()
    fblock.eval()

    t_err = _equivariance_error(tblock, x_r, x_i)
    f_err = _equivariance_error(fblock, x_r, x_i)

    print(f"ComplexEqTMambaBlock equivariance error: {t_err.item()}")
    print(f"ComplexEqFMambaBlock equivariance error: {f_err.item()}")
    assert t_err < 1e-3
    assert f_err < 1e-3


def test_gre_vss_mid_fusion_equivariance():
    device = _get_device()
    cfg = _get_cfg()
    hidden_dim = 48
    phase_channels = 24
    b, t, f = 2, 20, 17

    fusion = GREVSSMidFusion(
        hidden_dim=hidden_dim,
        phase_channels=phase_channels,
        cfg=cfg,
        d_state=cfg["model_cfg"].get("d_state", 16),
    ).to(device)
    fusion.eval()

    mag_in_fuse = torch.randn(b, hidden_dim, t, f, device=device)
    pha_abs_in_fuse = torch.randn(b, hidden_dim, t, f, device=device).abs()
    mag_gate_feat = torch.randn(b, hidden_dim, t, f, device=device)

    pha_res_r = torch.randn(b, phase_channels, t, f, device=device)
    pha_res_i = torch.randn(b, phase_channels, t, f, device=device)
    pha_feat_r = torch.randn(b, phase_channels, t, f, device=device)
    pha_feat_i = torch.randn(b, phase_channels, t, f, device=device)

    mag_fused_1, out_r_1, out_i_1 = fusion(
        mag_in_fuse=mag_in_fuse,
        pha_abs_in_fuse=pha_abs_in_fuse,
        mag_gate_feat=mag_gate_feat,
        pha_res_r=pha_res_r,
        pha_res_i=pha_res_i,
        pha_feat_r=pha_feat_r,
        pha_feat_i=pha_feat_i,
    )

    theta = torch.rand(1, device=device) * 2 * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    pha_res_r2 = cos_t * pha_res_r - sin_t * pha_res_i
    pha_res_i2 = sin_t * pha_res_r + cos_t * pha_res_i
    pha_feat_r2 = cos_t * pha_feat_r - sin_t * pha_feat_i
    pha_feat_i2 = sin_t * pha_feat_r + cos_t * pha_feat_i

    mag_fused_2, out_r_2, out_i_2 = fusion(
        mag_in_fuse=mag_in_fuse,
        pha_abs_in_fuse=pha_abs_in_fuse,
        mag_gate_feat=mag_gate_feat,
        pha_res_r=pha_res_r2,
        pha_res_i=pha_res_i2,
        pha_feat_r=pha_feat_r2,
        pha_feat_i=pha_feat_i2,
    )

    target_r = cos_t * out_r_1 - sin_t * out_i_1
    target_i = sin_t * out_r_1 + cos_t * out_i_1

    phase_err = (
        (out_r_2 - target_r).abs().mean()
        + (out_i_2 - target_i).abs().mean()
    ) / (
        target_r.abs().mean() + target_i.abs().mean() + 1e-8
    )
    mag_err = (mag_fused_2 - mag_fused_1).abs().mean() / (mag_fused_1.abs().mean() + 1e-8)

    print(f"GREVSSMidFusion phase equivariance error: {phase_err.item()}")
    print(f"GREVSSMidFusion mag invariant error: {mag_err.item()}")
    assert phase_err < 1e-3
    assert mag_err < 1e-3


def test_phase_mid_adapters_shape():
    device = _get_device()
    x = torch.randn(2, 24, 50, 33, device=device)
    to_complex = PhaseMidToComplex2D(24).to(device)
    to_real = PhaseMidToReal2D(24).to(device)

    x_r, x_i = to_complex(x)
    y = to_real(x_r, x_i)

    assert x_r.shape == x.shape
    assert x_i.shape == x.shape
    assert y.shape == x.shape
    assert torch.isfinite(x_r).all()
    assert torch.isfinite(x_i).all()
    assert torch.isfinite(y).all()


def test_full_model_construction():
    device = _get_device()
    cfg = _get_cfg()
    model = MambaSEUNet(cfg).to(device)

    assert model.enable_complex_phase_bottleneck
    assert isinstance(model.pha_TM_middle[0], ComplexEqTMambaBlock)
    assert isinstance(model.pha_FM_middle[0], ComplexEqFMambaBlock)
    assert isinstance(model.mid_fusions[0], GREVSSMidFusion)

