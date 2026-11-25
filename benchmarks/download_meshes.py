"""Download the real CFD meshes used by ``bench_real.py`` into ``benchmarks/data`` (not tracked).

* CFD-GCN (Belbute-Peres, Economon, Kolter, ICML 2020, https://github.com/locuslab/cfd-gcn):
  SU2 meshes around NACA0012 / NACA4412 / RAE2822 airfoils at several resolutions
  (332 to 14 576 nodes). The repository has no license file: the meshes are downloaded,
  never redistributed here.
* AirfRANS (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks, ODbL v1.0,
  https://airfrans.readthedocs.io): one RANS simulation around an airfoil, at its original
  OpenFOAM resolution (169 078 nodes, "clipped" record) and remeshed (8 304 nodes). Only the
  first sample of each 0.5 GB / 9 GB archive is streamed: a few tens of MB are downloaded.

    python benchmarks/download_meshes.py
"""
import os
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

CFDGCN = "https://raw.githubusercontent.com/locuslab/cfd-gcn/master/meshes/mesh_{}.su2"
CFDGCN_MESHES = ["NACA0012_xcoarse", "NACA0012_coarse", "NACA0012_fine",
                 "NACA4412_xcoarse", "NACA4412_lessfine", "NACA4412_fine",
                 "RAE2822_xcoarse", "RAE2822_fine"]
AIRFRANS = {
    "airfrans_remeshed": "https://zenodo.org/api/records/14840388/files/AirfRANS_remeshed.tar.gz/content",
    "airfrans_clipped": "https://zenodo.org/api/records/14840377/files/AirfRANS_clipped.tar.gz/content",
}
UA = {"User-Agent": "batched-splineconv-benchmark"}


def _safe_extract(tar, member, out):
    parts = member.name.split("/")
    if member.name.startswith(("/", "\\")) or ".." in parts or not (member.isfile() or member.isdir()):
        raise ValueError(f"refusing to extract {member.name!r}")
    tar.extract(member, out)


def stream_first_sample(url: str, out: str, depth: int = 3) -> str:
    """Extract only the first ``dataset/samples/<sample>`` directory of a remote .tar.gz."""
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        first = None
        for m in tar:
            parts = m.name.strip("./").split("/")
            if len(parts) < depth:
                continue
            sample = "/".join(parts[:depth])
            first = first or sample
            if sample != first:
                break
            if m.isfile():
                _safe_extract(tar, m, out)
    return first


def main():
    os.makedirs(os.path.join(DATA, "cfdgcn"), exist_ok=True)
    for name in CFDGCN_MESHES:
        dst = os.path.join(DATA, "cfdgcn", f"mesh_{name}.su2")
        if not os.path.exists(dst):
            req = urllib.request.Request(CFDGCN.format(name), headers=UA)
            with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
                f.write(r.read())
        print("ok", dst)
    for name, url in AIRFRANS.items():
        out = os.path.join(DATA, name)
        if not os.path.isdir(out):
            print("streaming first sample of", url)
            print("ok", os.path.join(out, stream_first_sample(url, out)))
        else:
            print("ok", out)


if __name__ == "__main__":
    main()
