"""Volume backends.

Anything with `.shape` (z, y, x) and numpy-style 3D slicing works as an inkdx
volume: numpy arrays, zarr arrays (local or fsspec-backed), or the
LayerStackVolume below for pre-extracted surface volumes stored as per-layer
TIFFs (`layers/00.tif ... NN.tif`), the format ink-detection segments ship in.

For a surface volume the "mesh" is implicit — the identity grid at the stack's
center layer; `identity_segment` builds it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile

from inkdx.io.segment import Segment

_LAYER_RE = re.compile(r"^(\d+)\.(tif|tiff)$")


class LayerStackVolume:
    """Lazy (n_layers, H, W) view over a directory of per-layer TIFFs.

    Layers are memory-mapped on first touch, so slicing reads only the pages
    and regions actually requested — a full-segment stack (tens of GB) costs
    nothing to open.
    """

    def __init__(self, layers_dir: str | Path) -> None:
        self.dir = Path(layers_dir)
        found: list[tuple[int, Path]] = []
        for p in self.dir.iterdir():
            m = _LAYER_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        if not found:
            raise FileNotFoundError(f"no NN.tif layers in {self.dir}")
        found.sort()
        indices = [i for i, _ in found]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"non-contiguous layer indices in {self.dir}: {indices}")
        self.paths = [p for _, p in found]
        self._mmaps: dict[int, np.ndarray] = {}

        first = self._layer(0)
        self.shape = (len(self.paths), *first.shape)
        self.dtype = first.dtype

    def _layer(self, k: int) -> np.ndarray:
        if k not in self._mmaps:
            self._mmaps[k] = tifffile.memmap(self.paths[k], mode="r")
        return self._mmaps[k]

    def __getitem__(self, key) -> np.ndarray:
        zk, yk, xk = key
        z_indices = range(*zk.indices(self.shape[0])) if isinstance(zk, slice) else [zk]
        planes = [self._layer(k)[yk, xk] for k in z_indices]
        out = np.stack(planes, axis=0)
        return out if isinstance(zk, slice) else out[0]


def open_surface_volume(path: str | Path, *, level: int = 0):
    """Open a surface volume: a layer-TIFF directory or an OME-Zarr store.

    OME-Zarr stores hold a multiscale pyramid as subgroups "0", "1", ...;
    `level` picks one (0 = full resolution). Plain zarr arrays work too.
    """
    path = Path(path)
    if path.suffix == ".zarr" or (path / ".zgroup").exists() or (path / ".zarray").exists():
        import zarr

        node = zarr.open(str(path), mode="r")
        if hasattr(node, "keys"):  # group: multiscale pyramid
            return node[str(level)]
        return node
    return LayerStackVolume(path)


def identity_segment(
    height: int,
    width: int,
    *,
    z_center: float,
    uuid: str = "surface-volume",
    valid: np.ndarray | None = None,
) -> Segment:
    """Implicit mesh of a surface volume: grid vertex (r, c) sits at
    (x=c, y=r, z=z_center) in the stack's own frame, normals along the layer
    axis. Use with LayerStackVolume so profiles read straight down the stack.
    """
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    z = np.full((height, width), float(z_center), dtype=np.float32)
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    return Segment(x=xx, y=yy, z=z, valid=valid.astype(bool), scale=(1.0, 1.0), uuid=uuid)
