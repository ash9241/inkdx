#!/usr/bin/env python
"""Run inkdx scan+surface diagnostics over a pre-extracted layer-stack segment.

A layer stack (`layers/00.tif ... NN.tif`) is a surface volume: the mesh is the
identity grid at the center layer and profiles read straight down the stack.

Example:
    python scripts/run_layer_stack.py /data/w00/layers out/w00_maps.npz \
        --mask /data/w00/w00_mask.png --processes 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from inkdx.io.volume import LayerStackVolume, identity_segment
from inkdx.runner import DiagnosticsConfig, run_diagnostics


def load_mask(path: str, shape_hw: tuple[int, int]) -> np.ndarray:
    if path.endswith((".tif", ".tiff")):
        import tifffile

        mask = tifffile.imread(path)
    else:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # gigapixel masks are normal here
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != shape_hw:
        raise SystemExit(f"mask shape {mask.shape} != stack plane {shape_hw}")
    return mask != 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layers_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--mask", default=None, help="validity mask (png/tif, nonzero=valid)")
    ap.add_argument("--z-center", type=float, default=None, help="default: stack center")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--halfwidth", type=int, default=None, help="default: max symmetric")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=0)
    ap.add_argument("--expected-thickness", type=float, default=12.0)
    args = ap.parse_args()

    stack = LayerStackVolume(args.layers_dir)
    nz, h, w = stack.shape
    zc = args.z_center if args.z_center is not None else (nz - 1) / 2.0
    hw = args.halfwidth if args.halfwidth is not None else int(min(zc, nz - 1 - zc))
    valid = load_mask(args.mask, (h, w)) if args.mask else None

    seg = identity_segment(h, w, z_center=zc, valid=valid)
    cfg = DiagnosticsConfig(
        tile_px=args.tile,
        halfwidth=hw,
        samples_per_tile=args.samples,
        seed=args.seed,
        processes=args.processes,
        expected_thickness=args.expected_thickness,
    )
    print(f"stack {stack.shape}, z_center {zc}, halfwidth {hw}, "
          f"tiles {-(-h // args.tile)}x{-(-w // args.tile)}, "
          f"processes {args.processes}", flush=True)

    t0 = time.time()
    maps = run_diagnostics(stack, seg, cfg, progress=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(args.out_npz, **maps)
    print(f"wrote {args.out_npz}")
    for k in ("cnr", "noise_sigma", "peak_offset", "peak_prominence", "haze_index"):
        m = maps[k]
        fin = m[np.isfinite(m)]
        if fin.size:
            print(f"  {k}: median {np.median(fin):.2f}  p5 {np.percentile(fin, 5):.2f}  "
                  f"p95 {np.percentile(fin, 95):.2f}  ({fin.size}/{m.size} tiles)")


if __name__ == "__main__":
    main()
