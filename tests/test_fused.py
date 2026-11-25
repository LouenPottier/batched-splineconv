"""The fused CUDA kernel against the SpMM path (itself checked against PyG in float64)."""
import pytest
import torch

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv import fused
from batched_splineconv.meshes import cartesian_pseudo, jittered_grid, radius_graph

pytestmark = pytest.mark.skipif(not fused.is_available(), reason="fused CUDA extension unavailable")
TOL = dict(rtol=1e-4, atol=1e-4)


def run(conv, x, op):
    x = x.clone().requires_grad_()
    out = conv(x, op)
    g = torch.ones_like(out).normal_(generator=torch.Generator(out.device).manual_seed(1))
    out.backward(g)
    grads = [x.grad] + [p.grad.clone() for p in conv.parameters()]
    conv.zero_grad()
    return out, grads


@pytest.mark.parametrize("dim,ks,degree", [(2, 5, 1), (2, 5, 2), (3, 3, 1), (3, 4, 3)])
@pytest.mark.parametrize("B,c_in,c_out", [(1, 3, 7), (4, 16, 16), (5, 70, 33), (3, 2, 130)])
def test_fused_matches_spmm(dim, ks, degree, B, c_in, c_out):
    torch.manual_seed(0)
    dev = "cuda"
    pos = torch.rand(300, dim, device=dev)
    ei = radius_graph(pos, None, 0.25 if dim == 2 else 0.4, p=2.0)
    op = SplineOperator(ei, cartesian_pseudo(pos, pos, ei, 0.4), ks, True, degree, 300).to(dev)

    ref = FixedMeshSplineConv(c_in, c_out, dim, ks, order="aggregate").to(dev)
    with torch.no_grad():
        ref.bias.normal_()
    fu = FixedMeshSplineConv(c_in, c_out, dim, ks, order="fused").to(dev)
    fu.load_state_dict(ref.state_dict())

    x = torch.randn(B, 300, c_in, device=dev)
    out_r, g_r = run(ref, x, op)
    out_f, g_f = run(fu, x, op)
    torch.testing.assert_close(out_f, out_r, **TOL)
    for a, b in zip(g_f, g_r):
        torch.testing.assert_close(a, b, **TOL)


def test_fused_bipartite_and_isolated_nodes():
    """Mesh-to-mesh operator where some destination nodes have no neighbour at all."""
    torch.manual_seed(0)
    dev = "cuda"
    ps, pd = torch.rand(200, 2, device=dev), torch.rand(90, 2, device=dev) * 1.5
    ei = radius_graph(ps, pd, 0.2, p=2.0)
    op = SplineOperator(ei, cartesian_pseudo(ps, pd, ei, 0.2), 5, num_nodes=(200, 90)).to(dev)
    ref = FixedMeshSplineConv(8, 12, 2, 5, root_weight=False, order="transform").to(dev)
    fu = FixedMeshSplineConv(8, 12, 2, 5, root_weight=False, order="fused").to(dev)
    fu.load_state_dict(ref.state_dict())
    x = torch.randn(3, 200, 8, device=dev)
    out_r, g_r = run(ref, x, op)
    out_f, g_f = run(fu, x, op)
    torch.testing.assert_close(out_f, out_r, **TOL)
    for a, b in zip(g_f, g_r):
        torch.testing.assert_close(a, b, **TOL)


def test_fused_deterministic():
    dev = "cuda"
    _, ei, pseudo = jittered_grid((40, 40), 2, device=dev)
    op = SplineOperator(ei, pseudo, 5, num_nodes=1600).to(dev)
    conv = FixedMeshSplineConv(16, 16, 2, 5, order="fused").to(dev)
    x = torch.randn(8, 1600, 16, device=dev)
    a, ga = run(conv, x, op)
    b, gb = run(conv, x, op)
    assert torch.equal(a, b) and all(torch.equal(u, v) for u, v in zip(ga, gb))
