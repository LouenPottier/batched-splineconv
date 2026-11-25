import pytest
import torch

from batched_splineconv.meshes import adaptive_radius_graph, grid_positions, rcm_permutation, read_su2

pytest.importorskip("scipy")

SU2 = """%
NDIME= 2
NELEM= 2
9 0 1 4 3 0
5 1 2 4 1
NPOIN= 5
0.0 0.0 0
1.0 0.0 1
2.0 0.0 2
0.0 1.0 3
1.0 1.0 4
NMARK= 0
"""


def test_read_su2(tmp_path):
    p = tmp_path / "m.su2"
    p.write_text(SU2)
    pos, tri = read_su2(str(p))
    assert pos.shape == (5, 2) and tri.shape == (3, 2 + 1)   # quad split in 2 + 1 triangle
    assert tri.max() == 4


def test_adaptive_radius_on_grid():
    """On a unit grid, h = 1 and R = (sqrt(2)/2) * 5 ~ 3.54: the disc of the report (fig. 4.3)."""
    pos = grid_positions((20, 20))
    ei, pseudo = adaptive_radius_graph(pos, None, 5)
    assert pseudo.min() >= 0 and pseudo.max() <= 1
    d = (pos[ei[0]] - pos[ei[1]]).norm(dim=1)
    assert d.max() <= 3.54 + 1e-4
    deg = torch.bincount(ei[1], minlength=400)
    assert deg.max() == 37   # interior nodes: lattice points in a disc of radius 3.54 (incl. self)


def test_rcm_is_permutation():
    pos = grid_positions((10, 10))
    ei, _ = adaptive_radius_graph(pos, None, 3)
    perm = rcm_permutation(ei, 100)
    assert torch.equal(perm.sort().values, torch.arange(100))


def test_hierarchy_preserves_refinement():
    """Decimating a graded point set keeps it graded: the dense half stays denser."""
    from batched_splineconv.meshes import mesh_hierarchy

    g = torch.Generator().manual_seed(0)
    dense = torch.rand(4000, 2, generator=g) * torch.tensor([0.5, 1.0])          # x in [0, 0.5]
    sparse = torch.rand(1000, 2, generator=g) * torch.tensor([0.5, 1.0]) + torch.tensor([0.5, 0.0])
    levels = mesh_hierarchy(torch.cat([dense, sparse]), 3, factor=3.0)
    sizes = [p.size(0) for p in levels]
    assert all(abs(sizes[i] / sizes[i + 1] - 3.0) < 0.5 for i in range(2)), sizes
    for p in levels[1:]:
        left = (p[:, 0] < 0.5).sum().item()
        assert left > 2 * (p.size(0) - left)   # still ~4x denser on the left
