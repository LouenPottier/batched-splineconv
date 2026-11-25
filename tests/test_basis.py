import pytest
import torch

from batched_splineconv import spline_basis

tsc = pytest.importorskip("torch_spline_conv")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("is_open", [True, False])
def test_basis_matches_torch_spline_conv(device, dim, degree, is_open):
    torch.manual_seed(0)
    pseudo = torch.rand(500, dim, device=device)
    pseudo[:10] = 0.0  # boundaries
    pseudo[10:20] = 1.0
    ks = torch.tensor([5, 4, 6][:dim], device=device)
    is_open_t = torch.tensor([is_open] * dim, dtype=torch.uint8, device=device)

    b_ref, wi_ref = tsc.spline_basis(pseudo, ks, is_open_t, degree)
    b, wi = spline_basis(pseudo, ks, is_open, degree)

    torch.testing.assert_close(b, b_ref, rtol=1e-5, atol=1e-6)
    # where the basis value is exactly zero the index is irrelevant (and may differ)
    assert torch.equal(wi[b_ref != 0], wi_ref[b_ref != 0])
