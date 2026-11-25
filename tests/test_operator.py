import pytest
import torch

from batched_splineconv import FixedMeshSplineConv, SplineOperator
from batched_splineconv.meshes import cartesian_pseudo, radius_graph


def make_op():
    torch.manual_seed(0)
    pos = torch.rand(120, 2)
    ei = radius_graph(pos, None, 0.3, p=2.0)
    return SplineOperator(ei, cartesian_pseudo(pos, pos, ei, 0.3), 5, num_nodes=120)


def test_layouts_are_lazy():
    op = make_op()
    assert op._cache == {}
    _ = op.A
    assert set(op._cache) == {"A"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_moving_the_operator_rebuilds_layouts():
    op = make_op()
    conv = FixedMeshSplineConv(3, 4, 2, 5, order="aggregate")
    x = torch.randn(2, 120, 3)
    ref = conv(x, op)                      # builds layouts on CPU
    op, conv = op.cuda(), conv.cuda()
    assert op._cache == {}                 # invalidated by the move
    out = conv(x.cuda(), op)
    assert op.A.device.type == "cuda"
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-6)
    op64 = op.to(torch.float64)
    assert op64.A.dtype == torch.float64
