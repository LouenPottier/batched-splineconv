"""On a regular grid, SplineConv *is* a discrete convolution.

With degree-1 splines, ``kernel_size = 2r+1`` and pseudo-coordinates ``u = d/(2r) + 1/2``,
every grid offset ``d in {-r..r}^D`` lands exactly on a spline knot: the basis is one-hot
and ``h_theta(d) = theta_d``. Each offset owns exactly one trainable weight, as in a
discrete kernel, so the layer reproduces ``conv{2,3}d`` / ``conv_transpose2d`` exactly,
with the same parameter count and the same dependence on its parameters.
"""
import pytest
import torch
import torch.nn.functional as F

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import cartesian_pseudo, grid_positions, stencil_edges

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = dict(rtol=1e-4, atol=1e-5)


def grid_operator(src_shape, dst_shape, r, stride, device):
    pos_src = stride * grid_positions(src_shape, device=device)
    pos_dst = grid_positions(dst_shape, device=device)
    ei = stencil_edges(src_shape, dst_shape, r, stride=stride, device=device)
    pseudo = cartesian_pseudo(pos_src, pos_dst, ei, r)
    n_src, n_dst = pos_src.size(0), pos_dst.size(0)
    return SplineOperator(ei, pseudo, 2 * r + 1, True, 1, (n_src, n_dst), aggr="add").to(device)


def to_nodes(img):   # [B, C, *spatial] -> [B, N, C]
    return img.flatten(2).transpose(1, 2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ["aggregate", "transform"])
@pytest.mark.parametrize("r", [1, 2])
def test_conv2d(device, order, r):
    torch.manual_seed(0)
    B, c_in, c_out, H, W = 3, 4, 5, 12, 9
    k = 2 * r + 1
    img = torch.randn(B, c_in, H, W, device=device)
    w = torch.randn(c_out, c_in, k, k, device=device)

    conv = FixedMeshSplineConv(c_in, c_out, 2, k, root_weight=False, bias=False, order=order).to(device)
    with torch.no_grad():  # one spline weight per grid offset: theta_d = w_d
        conv.weight.copy_(w.permute(2, 3, 1, 0).reshape(k * k, c_in, c_out))
    op = grid_operator((H, W), (H, W), r, 1, device)

    ref = F.conv2d(img, w, padding=r)
    torch.testing.assert_close(conv(to_nodes(img), op), to_nodes(ref), **TOL)


@pytest.mark.parametrize("device", DEVICES)
def test_conv3d(device):
    torch.manual_seed(0)
    B, c_in, c_out, shape = 2, 3, 4, (5, 6, 7)
    img = torch.randn(B, c_in, *shape, device=device)
    w = torch.randn(c_out, c_in, 3, 3, 3, device=device)

    conv = FixedMeshSplineConv(c_in, c_out, 3, 3, root_weight=False, bias=False).to(device)
    with torch.no_grad():
        conv.weight.copy_(w.permute(2, 3, 4, 1, 0).reshape(27, c_in, c_out))
    op = grid_operator(shape, shape, 1, 1, device)

    ref = F.conv3d(img, w, padding=1)
    torch.testing.assert_close(conv(to_nodes(img), op), to_nodes(ref), **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ["aggregate", "transform"])
def test_conv_transpose2d_upsampling(device, order):
    """Coarse grid -> fine grid (stride 2): the mesh-to-mesh layer of a decoder."""
    torch.manual_seed(0)
    B, c_in, c_out, h, w_ = 2, 4, 3, 6, 5
    img = torch.randn(B, c_in, h, w_, device=device)
    w = torch.randn(c_in, c_out, 5, 5, device=device)

    conv = FixedMeshSplineConv(c_in, c_out, 2, 5, root_weight=False, bias=False, order=order).to(device)
    with torch.no_grad():  # transposed conv indexes its kernel by dst - src: flip it
        conv.weight.copy_(w.flip(2, 3).permute(2, 3, 0, 1).reshape(25, c_in, c_out))
    op = grid_operator((h, w_), (2 * h, 2 * w_), 2, 2, device)

    ref = F.conv_transpose2d(img, w, stride=2, padding=2, output_padding=1)
    assert ref.shape[-2:] == (2 * h, 2 * w_)
    torch.testing.assert_close(conv(to_nodes(img), op), to_nodes(ref), **TOL)


@pytest.mark.parametrize("device", DEVICES)
def test_smooth_under_mesh_perturbation(device):
    """The point of the continuous kernel: moving nodes a little changes the output a little.

    Output difference grows linearly with the jitter amplitude (Lipschitz in node positions),
    instead of jumping as it would with a nearest-offset lookup of a discrete kernel.
    """
    torch.manual_seed(0)
    H = W = 16
    r = 2
    pos = grid_positions((H, W), device=device)
    ei = stencil_edges((H, W), (H, W), r, device=device)
    conv = FixedMeshSplineConv(2, 3, 2, 2 * r + 1, root_weight=False, bias=False).to(device)
    x = torch.randn(4, H * W, 2, device=device)
    noise = torch.rand_like(pos) - 0.5

    def run(eps):
        p = pos + eps * noise
        op = SplineOperator(ei, cartesian_pseudo(p, p, ei, r + 0.5), 2 * r + 1, num_nodes=H * W).to(device)
        return conv(x, op)

    base = run(0.0)
    diffs = torch.stack([(run(e) - base).abs().max() for e in (0.01, 0.02, 0.04)])
    ratios = diffs[1:] / diffs[:-1]
    assert torch.allclose(ratios, torch.full_like(ratios, 2.0), rtol=0.15), diffs
