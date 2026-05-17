import torch

from models.generator import PhaFFNAdapter2D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = 16
    module = PhaFFNAdapter2D(channels=channels, dropout=0.0, res_scale=0.1).to(device)
    x = torch.randn(2, channels, 100, 129, device=device)
    y = module(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "non-finite values in output"
    loss = y.abs().mean()
    loss.backward()
    print("PhaFFNAdapter2D sanity check passed on", device)


if __name__ == "__main__":
    main()

