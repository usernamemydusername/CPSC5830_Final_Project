#!/usr/bin/env python3
"""
Build a simple region-to-coordinate-prior dictionary from a processed Porto tar.gz bundle.

Output:
  region_coord_priors.pt

The saved object is a dictionary keyed by compact region id:

  priors[region_id] = {
      "count": n,
      "centroid": {"x": ..., "y": ..., "lon": ..., "lat": ...},
      "mean": {"offset_x": ..., "offset_y": ..., "x": ..., "y": ..., "lon": ..., "lat": ...},
      "shrink": {
          0:  {"offset_x": ..., "offset_y": ..., "x": ..., "y": ..., "lon": ..., "lat": ...},
          5:  {...},
          10: {...},
          20: {...},
      },
  }

Notes:
  - All statistics are computed from the TRAIN split only.
  - alpha = 0 is exactly the empirical mean offset.
  - For regions with no train destinations, mean/shrink offsets are set to 0,
    so the predicted coordinate falls back to the region centroid.
"""

from __future__ import annotations

import argparse
import pickle
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from pyproj import Transformer


def extract_bundle(bundle_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    children = list(extract_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def iter_train_shards(data_dir: Path) -> Iterable[Path]:
    shard_dir = data_dir / "supervised_shards" / "train"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Could not find train shard directory: {shard_dir}")
    yield from sorted(shard_dir.glob("train_*.pt"))


def load_mappings(data_dir: Path) -> dict:
    with open(data_dir / "id_mappings.pkl", "rb") as f:
        return pickle.load(f)


def compute_centroids(mappings: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid_x and centroid_y arrays indexed by compact region id."""
    region_id_map: Dict[int, int] = mappings["region_id_map"]
    grid = mappings["grid"]

    n_regions = len(region_id_map)
    centroid_x = np.zeros(n_regions, dtype=np.float64)
    centroid_y = np.zeros(n_regions, dtype=np.float64)

    xmin = float(grid["xmin"])
    ymin = float(grid["ymin"])
    n_cols = int(grid["n_cols"])
    cell_size = float(grid["cell_size"])

    for raw_region_id, compact_region_id in region_id_map.items():
        raw_region_id = int(raw_region_id)
        compact_region_id = int(compact_region_id)

        row = raw_region_id // n_cols
        col = raw_region_id % n_cols

        centroid_x[compact_region_id] = xmin + (col + 0.5) * cell_size
        centroid_y[compact_region_id] = ymin + (row + 0.5) * cell_size

    return centroid_x, centroid_y


def collect_train_offsets(data_dir: Path, n_regions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return count, sum_offset_x, sum_offset_y from train split."""
    count = np.zeros(n_regions, dtype=np.int64)
    sum_dx = np.zeros(n_regions, dtype=np.float64)
    sum_dy = np.zeros(n_regions, dtype=np.float64)

    total = 0
    for shard_path in iter_train_shards(data_dir):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for ex in shard:
            r = int(ex["dest_region"])
            count[r] += 1
            sum_dx[r] += float(ex["dest_offset_x"])
            sum_dy[r] += float(ex["dest_offset_y"])
            total += 1
        print(f"Loaded {shard_path.name}; cumulative train examples = {total:,}", flush=True)

    return count, sum_dx, sum_dy


def xy_to_lonlat(transformer: Transformer, x: float, y: float) -> Tuple[float, float]:
    lon, lat = transformer.transform(float(x), float(y))
    return float(lon), float(lat)


def build_prior_dict(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    count: np.ndarray,
    sum_dx: np.ndarray,
    sum_dy: np.ndarray,
    alphas: List[float],
) -> dict:
    transformer = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
    priors = {}

    for r in range(len(count)):
        n = int(count[r])

        if n > 0:
            mean_dx = float(sum_dx[r] / n)
            mean_dy = float(sum_dy[r] / n)
        else:
            mean_dx = 0.0
            mean_dy = 0.0

        cx = float(centroid_x[r])
        cy = float(centroid_y[r])
        clon, clat = xy_to_lonlat(transformer, cx, cy)

        mean_x = cx + mean_dx
        mean_y = cy + mean_dy
        mean_lon, mean_lat = xy_to_lonlat(transformer, mean_x, mean_y)

        region_entry = {
            "count": n,
            "centroid": {
                "x": cx,
                "y": cy,
                "lon": clon,
                "lat": clat,
            },
            "mean": {
                "offset_x": mean_dx,
                "offset_y": mean_dy,
                "x": mean_x,
                "y": mean_y,
                "lon": mean_lon,
                "lat": mean_lat,
            },
            "shrink": {},
        }

        for alpha in alphas:
            if alpha == 0:
                weight = 1.0 if n > 0 else 0.0
            else:
                weight = n / (n + alpha) if n > 0 else 0.0

            sdx = weight * mean_dx
            sdy = weight * mean_dy
            sx = cx + sdx
            sy = cy + sdy
            slon, slat = xy_to_lonlat(transformer, sx, sy)

            alpha_key = int(alpha) if float(alpha).is_integer() else float(alpha)
            region_entry["shrink"][alpha_key] = {
                "offset_x": float(sdx),
                "offset_y": float(sdy),
                "x": float(sx),
                "y": float(sy),
                "lon": float(slon),
                "lat": float(slat),
            }

        priors[int(r)] = region_entry

    return priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True, help="Path to porto_data_bundle_*.tar.gz.")
    parser.add_argument("--out", type=Path, default=Path("region_coord_priors.pt"), help="Output .pt dictionary path.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0, 5, 10, 20],
        help="Shrinkage alpha values. alpha=0 is the same as the empirical mean offset.",
    )
    parser.add_argument("--extract-dir", type=Path, default=None, help="Optional directory to extract bundle.")
    args = parser.parse_args()

    if args.extract_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = extract_bundle(args.bundle, Path(tmp))
            mappings = load_mappings(data_dir)
            centroid_x, centroid_y = compute_centroids(mappings)
            count, sum_dx, sum_dy = collect_train_offsets(data_dir, len(centroid_x))
            priors = build_prior_dict(centroid_x, centroid_y, count, sum_dx, sum_dy, args.alphas)
    else:
        data_dir = extract_bundle(args.bundle, args.extract_dir)
        mappings = load_mappings(data_dir)
        centroid_x, centroid_y = compute_centroids(mappings)
        count, sum_dx, sum_dy = collect_train_offsets(data_dir, len(centroid_x))
        priors = build_prior_dict(centroid_x, centroid_y, count, sum_dx, sum_dy, args.alphas)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(priors, args.out)
    print(f"Wrote {args.out}")
    print("Example usage:")
    print("  priors = torch.load('region_coord_priors.pt', weights_only=False)")
    print("  coord = priors[pred_region]['shrink'][10]  # coord['lon'], coord['lat']")


if __name__ == "__main__":
    main()
