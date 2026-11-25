"""Equivalence with PyG's SplineConv applied to a batch seen as B disjoint copies of the mesh."""
import warnings

import pytest
import torch

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import cartesian_pseudo, radius_graph

pyg_nn = pytest.importorskip("torch_geometric.nn")
pytest.importorskip("torch_spline_conv")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# float64 so that agreement is checked to ~1e-10, not to fp32 accumulation noise
TOL = dict(rtol=1e-9, atol=1e-10)
DT = torch.float64


def random_mesh(n_src, n_dst, dim, radius, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos_src = torch.rand(n_src, dim, generator=g, dtype=DT).to(device)
    pos_dst = pos_src if n_dst is None else torch.rand(n_dst, dim, generator=g, dtype=DT).to(device)
    ei = radius_graph(pos_src, pos_dst, radius, p=2.0)
    return ei, cartesian_pseudo(pos_src, pos_dst, ei, radius)


def batched_disjoint(edge_index, pseudo, B, n_src, n_dst):
    """The usual PyG batch: B copies of the graph with shifted node indices."""
    shift = torch.arange(B, device=edge_index.device)
    src = (edge_index[0][None] + n_src * shift[:, None]).reshape(-1)
    dst = (edge_index[1][None] + n_dst * shift[:, None]).reshape(-1)
    return torch.stack([src, dst]), pseudo.repeat(B, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim,kernel_size", [(2, 5), (3, 3)])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("is_open", [True, False])
@pytest.mark.parametrize("aggr", ["mean", "add"])
@pytest.mark.parametrize("order", ["aggregate", "transform"])
def test_same_mesh(device, dim, kernel_size, degree, is_open, aggr, order):
    torch.manual_seed(0)
    N, B, c_in, c_out = 80, 3, 4, 6
    ei, pseudo = random_mesh(N, None, dim, 0.35, device)

    ref = pyg_nn.SplineConv(c_in, c_out, dim, kernel_size, is_open, degree, aggr=aggr).to(device, DT)
    with torch.no_grad():
        ref.bias.normal_()
    conv = FixedMeshSplineConv.from_pyg(ref, order=order)
    op = SplineOperator(ei, pseudo, kernel_size, is_open, degree, N, aggr).to(device)

    x = torch.randn(B, N, c_in, device=device, dtype=DT, requires_grad=True)
    x_ref = x.detach().reshape(B * N, c_in).requires_grad_()
    ei_b, ps_b = batched_disjoint(ei, pseudo, B, N, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_ref = ref(x_ref, ei_b, ps_b).view(B, N, c_out)
    out = conv(x, op)
    torch.testing.assert_close(out, out_ref, **TOL)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    torch.testing.assert_close(x.grad, x_ref.grad.view(B, N, c_in), **TOL)
    torch.testing.assert_close(conv.weight.grad, ref.weight.grad, **TOL)
    torch.testing.assert_close(conv.root.grad, ref.lin.weight.grad.t(), **TOL)
    torch.testing.assert_close(conv.bias.grad, ref.bias.grad, **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ["aggregate", "transform"])
def test_bipartite_mesh(device, order):
    """Mesh-to-mesh layer (pooling / unpooling between two different meshes)."""
    torch.manual_seed(0)
    n_src, n_dst, B, c_in, c_out, dim = 90, 40, 2, 3, 5, 2
    ei, pseudo = random_mesh(n_src, n_dst, dim, 0.3, device)

    ref = pyg_nn.SplineConv((c_in, c_in), c_out, dim, 5).to(device, DT)
    conv = FixedMeshSplineConv.from_pyg(ref, order=order)
    op = SplineOperator(ei, pseudo, 5, num_nodes=(n_src, n_dst)).to(device)

    x = torch.randn(B, n_src, c_in, device=device, dtype=DT)
    x_dst = torch.randn(B, n_dst, c_in, device=device, dtype=DT)
    ei_b, ps_b = batched_disjoint(ei, pseudo, B, n_src, n_dst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_ref = ref((x.reshape(-1, c_in), x_dst.reshape(-1, c_in)), ei_b, ps_b,
                      size=(B * n_src, B * n_dst)).view(B, n_dst, c_out)
    torch.testing.assert_close(conv(x, op, x_dst=x_dst), out_ref, **TOL)


@pytest.mark.parametrize("device", DEVICES)
def test_nbc_layout_and_stacking(device):
    torch.manual_seed(0)
    N, B = 50, 4
    ei, pseudo = random_mesh(N, None, 2, 0.4, device)
    op = SplineOperator(ei, pseudo, 5, num_nodes=N).to(device)
    l1 = FixedMeshSplineConv(3, 8, 2, 5).to(device, DT)
    l2 = FixedMeshSplineConv(8, 2, 2, 5).to(device, DT)
    x = torch.randn(B, N, 3, device=device, dtype=DT)
    a = l2(torch.relu(l1(x, op)), op)
    b = l2(torch.relu(l1(x.transpose(0, 1), op, layout="NBC")), op, layout="NBC").transpose(0, 1)
    torch.testing.assert_close(a, b)
