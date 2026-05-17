import math

import pytest
import torch

from models.mamba_block import EqMamba1D, EqTMambaBlock, EqFMambaBlock


def _get_cfg():
    return {
        "model_cfg": {
            "d_state": 16,
            "d_conv": 4,
            "expand": 4,
            "norm_epsilon": 1e-5,
            "eq_mamba_res_scale": 1.0,
            "eq_mamba_dropout": 0.0,
            "eq_mamba_bidirectional": True,
            "eq_mamba_use_complex_conv": True,
        }
    }


def _get_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Eq-Mamba-v1 tests.")
    return torch.device("cuda")


def test_eq_mamba1d_shape():
    device = _get_device()
    cfg = _get_cfg()
    c = 24
    b = 2
    l = 64
    x_r = torch.randn(b, l, c, device=device)
    x_i = torch.randn(b, l, c, device=device)
    block = EqMamba1D(c, cfg).to(device)
    y_r, y_i = block(x_r, x_i)
    assert y_r.shape == x_r.shape
    assert y_i.shape == x_i.shape
    assert torch.isfinite(y_r).all()
    assert torch.isfinite(y_i).all()


def test_eq_mamba1d_equivariance():
    device = _get_device()
    cfg = _get_cfg()
    c = 24
    b = 2
    l = 64
    x_r = torch.randn(b, l, c, device=device)
    x_i = torch.randn(b, l, c, device=device)
    block = EqMamba1D(c, cfg).to(device)
    block.eval()

    theta = torch.rand(1, device=device) * 2 * math.pi
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

    print(f"EqMamba1D equivariance error: {err.item()}")
    assert err < 1e-3


def test_eq_tm_fm_mamba_block_shape():
    device = _get_device()
    cfg = _get_cfg()
    x = torch.randn(2, 24, 50, 33, device=device)
    tblock = EqTMambaBlock(cfg, 24).to(device)
    fblock = EqFMambaBlock(cfg, 24).to(device)
    yt = tblock(x)
    yf = fblock(x)
    assert yt.shape == x.shape
    assert yf.shape == x.shape
    assert torch.isfinite(yt).all()
    assert torch.isfinite(yf).all()


def test_eq_tm_fm_mamba_block_backward():
    device = _get_device()
    cfg = _get_cfg()
    x = torch.randn(2, 24, 50, 33, device=device)
    tblock = EqTMambaBlock(cfg, 24).to(device)
    fblock = EqFMambaBlock(cfg, 24).to(device)
    yt = tblock(x)
    yf = fblock(x)
    loss = yt.abs().mean() + yf.abs().mean()
    loss.backward()

