import math

import pytest
import torch

from models.generator import ComplexPhaFFN1D, PhaFFNAdapter2D


def _get_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Pha-FFN tests.")
    return torch.device("cuda")




def test_complex_pha_ffn1d_shape_backward():
    device = _get_device()
    c = 8
    b = 2
    l = 128
    x_r = torch.randn(b, l, c, device=device)
    x_i = torch.randn(b, l, c, device=device)

    ffn = ComplexPhaFFN1D(
        channels=c,
        expansion=4,
        dropout=0.0,
        norm_epsilon=1e-5,
        use_complex_conv=True,
        kernel_size=3,
    ).to(device)

    y_r, y_i = ffn(x_r, x_i)
    assert y_r.shape == x_r.shape
    assert y_i.shape == x_i.shape
    assert torch.isfinite(y_r).all()
    assert torch.isfinite(y_i).all()

    loss = y_r.abs().mean() + y_i.abs().mean()
    loss.backward()


def test_complex_pha_ffn1d_equivariance():
    device = _get_device()
    c = 8
    b = 2
    l = 128
    x_r = torch.randn(b, l, c, device=device)
    x_i = torch.randn(b, l, c, device=device)

    ffn = ComplexPhaFFN1D(
        channels=c,
        expansion=4,
        dropout=0.0,
        norm_epsilon=1e-5,
        use_complex_conv=True,
        kernel_size=3,
    ).to(device)
    ffn.eval()

    theta = torch.rand(1, device=device) * 2 * math.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x2_r = cos_t * x_r - sin_t * x_i
    x2_i = sin_t * x_r + cos_t * x_i

    y1_r, y1_i = ffn(x_r, x_i)
    y2_r, y2_i = ffn(x2_r, x2_i)

    target_r = cos_t * y1_r - sin_t * y1_i
    target_i = sin_t * y1_r + cos_t * y1_i

    err = (
        (y2_r - target_r).abs().mean()
        + (y2_i - target_i).abs().mean()
    ) / (
        target_r.abs().mean() + target_i.abs().mean() + 1e-8
    )

    print(f"ComplexPhaFFN1D equivariance error: {err.item()}")
    assert err < 1e-3


def test_pha_ffn_adapter2d_shape_backward():
    device = _get_device()
    x = torch.randn(2, 8, 100, 129, device=device)

    adapter = PhaFFNAdapter2D(
        channels=8,
        dropout=0.0,
        res_scale=1.0,
        expansion=4,
        norm_epsilon=1e-5,
        use_complex_conv=True,
        kernel_size=3,
    ).to(device)

    y = adapter(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.abs().mean()
    loss.backward()



