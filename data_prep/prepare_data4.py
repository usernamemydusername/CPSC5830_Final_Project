#!/usr/bin/env python3
"""
Prepare Porto taxi destination-prediction data with a heterogeneous urban graph.

Trial 3 changes:
  - Default output directory is data/trial3.
  - Each supervised trajectory produces multiple prefix examples.
  - Supervised examples are lightweight and do not duplicate full future trajectories.
  - POI features use semantic multi-hot OSM tags, not only four coarse flags.
  - Historical OSM snapshots are supported through --osm-date.
  - Polygon/multipolygon OSM POIs are converted to representative points before region joining.
  - Supervised examples are saved as shards instead of one huge .pt file.
  - Feature names and a tar.gz bundle are written for sharing.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import random
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

try:
    import geopandas as gpd
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires geopandas. Please run it in the cluster env with geopandas installed.") from e

try:
    import osmnx as ox
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires osmnx. Please run it in the cluster env with osmnx installed.") from e

try:
    import torch
    from torch_geometric.data import HeteroData
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires torch and torch_geometric for HeteroData output.") from e


@dataclass
class GridSpec:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cell_size: float
    n_cols: int
    n_rows: int


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_polyline(s: object) -> List[Tuple[float, float]]:
    """Return [(lon, lat), ...] from Kaggle POLYLINE string."""
    if pd.isna(s):
        return []
    try:
        pts = ast.literal_eval(str(s))
        out = []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            lon, lat = float(p[0]), float(p[1])
            if math.isfinite(lon) and math.isfinite(lat):
                out.append((lon, lat))
        return out
    except Exception:
        return []


def filter_bbox(points: Sequence[Tuple[float, float]], bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return [(lon, lat) for lon, lat in points if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat]


def iter_csv_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def collect_bounds(
    csv_paths: Sequence[Path],
    transformer: Transformer,
    bbox: Tuple[float, float, float, float],
    chunksize: int,
    sample_rows: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Collect projected bounds from valid GPS points, using bbox to remove long-distance anomalies."""
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    seen_rows = 0
    for path in csv_paths:
        if not path.exists():
            continue
        log(f"Scanning bounds from {path}")
        for chunk in iter_csv_chunks(path, chunksize):
            if sample_rows is not None and seen_rows >= sample_rows:
                break
            if sample_rows is not None:
                chunk = chunk.iloc[: max(0, sample_rows - seen_rows)]
            seen_rows += len(chunk)
            for s in chunk["POLYLINE"]:
                pts = filter_bbox(parse_polyline(s), bbox)
                if not pts:
                    continue
                lon = np.array([p[0] for p in pts], dtype=float)
                lat = np.array([p[1] for p in pts], dtype=float)
                x, y = transformer.transform(lon, lat)
                if len(x):
                    xs_min.append(float(np.min(x)))
                    xs_max.append(float(np.max(x)))
                    ys_min.append(float(np.min(y)))
                    ys_max.append(float(np.max(y)))
        if sample_rows is not None and seen_rows >= sample_rows:
            break
    if not xs_min:
        raise ValueError("No valid GPS points found. Check paths and bbox.")
    return min(xs_min), min(ys_min), max(xs_max), max(ys_max)


def make_grid(bounds: Tuple[float, float, float, float], cell_size: float, buffer_m: float) -> GridSpec:
    xmin, ymin, xmax, ymax = bounds
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    n_cols = int(math.ceil((xmax - xmin) / cell_size))
    n_rows = int(math.ceil((ymax - ymin) / cell_size))
    return GridSpec(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, cell_size=cell_size, n_cols=n_cols, n_rows=n_rows)


def xy_to_region_id(x: np.ndarray, y: np.ndarray, grid: GridSpec) -> np.ndarray:
    col = np.floor((x - grid.xmin) / grid.cell_size).astype(np.int64)
    row = np.floor((y - grid.ymin) / grid.cell_size).astype(np.int64)
    valid = (col >= 0) & (col < grid.n_cols) & (row >= 0) & (row < grid.n_rows)
    rid = row * grid.n_cols + col
    rid[~valid] = -1
    return rid


def compress_consecutive(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    last = None
    for v in seq:
        if v < 0:
            continue
        if last is None or v != last:
            out.append(int(v))
            last = v
    return out


def make_regions_gdf(grid: GridSpec, crs: str) -> gpd.GeoDataFrame:
    polys = []
    ids = []
    xs = []
    ys = []
    for row in range(grid.n_rows):
        y0 = grid.ymin + row * grid.cell_size
        for col in range(grid.n_cols):
            x0 = grid.xmin + col * grid.cell_size
            rid = row * grid.n_cols + col
            geom = box(x0, y0, x0 + grid.cell_size, y0 + grid.cell_size)
            polys.append(geom)
            ids.append(rid)
            c = geom.centroid
            xs.append(c.x)
            ys.append(c.y)
    return gpd.GeoDataFrame({"region_id": ids, "centroid_x": xs, "centroid_y": ys}, geometry=polys, crs=crs)


def split_name(i: int, n: int, train_frac: float, val_frac: float) -> str:
    if i < int(n * train_frac):
        return "train"
    if i < int(n * (train_frac + val_frac)):
        return "val"
    return "test"


def sample_prefix_lengths(num_points: int, num_prefix_samples: int, rng: random.Random) -> List[int]:
    """Uniformly sample prefix lengths from [2, T-1]. If too short, use all available lengths."""
    available = list(range(2, num_points))  # 2, ..., T-1
    if len(available) <= num_prefix_samples:
        return available
    return sorted(rng.sample(available, num_prefix_samples))


def safe_scalar(row: pd.Series, col: str):
    val = row.get(col)
    return None if pd.isna(val) else val


def _clear_pt_files(root: Path) -> None:
    """Remove stale .pt shard files below root."""
    if root.exists():
        for old in root.rglob("*.pt"):
            old.unlink()
    root.mkdir(parents=True, exist_ok=True)


def _flush_raw_buffer(
    buffers: Dict[str, List[dict]],
    raw_root: Path,
    raw_shard_counts: Dict[str, int],
    example_counts: Dict[str, int],
    split: str,
) -> None:
    """Write one raw shard for a split and clear its in-memory buffer."""
    if not buffers[split]:
        return
    split_dir = raw_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_id = raw_shard_counts[split]
    path = split_dir / f"{split}_raw_{shard_id:03d}.pt"
    torch.save(buffers[split], path)
    raw_shard_counts[split] += 1
    example_counts[split] += len(buffers[split])
    log(f"Saved raw {split} shard {shard_id:03d} with {len(buffers[split]):,} examples")
    buffers[split].clear()


def build_examples_and_transition_counts(
    train_path: Path,
    transformer: Transformer,
    bbox: Tuple[float, float, float, float],
    grid: GridSpec,
    regions_lookup: pd.DataFrame,
    chunksize: int,
    train_frac: float,
    val_frac: float,
    num_prefix_samples: int,
    min_points: int,
    seed: int,
    raw_shard_root: Path,
    shard_size: int,
    max_train_rows: Optional[int] = None,
) -> Tuple[Dict[str, int], Counter, Counter, Counter, Counter, set]:
    """Build lightweight raw supervised shards and transition counts.

    Each valid trip creates up to num_prefix_samples examples. The raw shards are
    intentionally lightweight: they keep the prefix and destination label, but do
    not store full_lonlat, prefix_lonlat, or the full future trajectory. This avoids
    repeating complete trajectories for every prefix example.
    """
    log("Loading train.csv metadata for deterministic time split...")
    usecols = ["TRIP_ID", "TIMESTAMP", "POLYLINE", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID", "DAY_TYPE", "MISSING_DATA"]
    meta = pd.read_csv(train_path, usecols=lambda c: c in usecols, nrows=max_train_rows)
    if "MISSING_DATA" in meta.columns:
        meta = meta[~meta["MISSING_DATA"].astype(bool)].copy()
    meta = meta.sort_values("TIMESTAMP").reset_index(drop=True)
    n = len(meta)
    log(f"Usable supervised trips after missing-data filter: {n:,}")

    _clear_pt_files(raw_shard_root)
    buffers: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    raw_shard_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    example_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    transition_counter: Counter = Counter()
    out_counter: Counter = Counter()
    in_counter: Counter = Counter()
    dest_counter: Counter = Counter()
    active_region_ids: set = set()

    rng = random.Random(seed)
    region_centroid = regions_lookup.set_index("region_id")[["centroid_x", "centroid_y"]]

    for i, row in meta.iterrows():
        pts = filter_bbox(parse_polyline(row["POLYLINE"]), bbox)
        if len(pts) < min_points:
            continue

        lon = np.array([p[0] for p in pts], dtype=float)
        lat = np.array([p[1] for p in pts], dtype=float)
        x, y = transformer.transform(lon, lat)
        region_raw = xy_to_region_id(np.asarray(x), np.asarray(y), grid)
        if region_raw[-1] < 0:
            continue

        full_region_seq_raw = compress_consecutive(region_raw.tolist())
        if len(full_region_seq_raw) < 2:
            continue

        # Count full-trajectory transitions once per trip, not once per prefix example.
        # We use these counts to build graph edges, but we do not store the full trajectory
        # inside every supervised example.
        for a, b in zip(full_region_seq_raw[:-1], full_region_seq_raw[1:]):
            if a != b:
                transition_counter[(a, b)] += 1
                out_counter[a] += 1
                in_counter[b] += 1
                active_region_ids.add(int(a))
                active_region_ids.add(int(b))

        dest_region = int(region_raw[-1])
        if dest_region not in region_centroid.index:
            continue
        dest_counter[dest_region] += 1
        active_region_ids.add(dest_region)

        dest_x = float(x[-1])
        dest_y = float(y[-1])
        cx = float(region_centroid.loc[dest_region, "centroid_x"])
        cy = float(region_centroid.loc[dest_region, "centroid_y"])
        offset_x = dest_x - cx
        offset_y = dest_y - cy

        prefix_lengths = sample_prefix_lengths(len(pts), num_prefix_samples, rng)
        if not prefix_lengths:
            continue

        sp = split_name(i, n, train_frac, val_frac)
        seen_prefixes = set()

        for sample_id, pref_len in enumerate(prefix_lengths):
            prefix_region_seq_raw = compress_consecutive(region_raw[:pref_len].tolist())
            if not prefix_region_seq_raw:
                continue
            key = tuple(prefix_region_seq_raw)
            if key in seen_prefixes:
                continue
            seen_prefixes.add(key)
            active_region_ids.update(int(r) for r in prefix_region_seq_raw)

            buffers[sp].append(
                {
                    "trip_id": row.get("TRIP_ID"),
                    "timestamp": int(row.get("TIMESTAMP")) if not pd.isna(row.get("TIMESTAMP")) else None,
                    "prefix_sample_id": int(sample_id),
                    "prefix_len_points": int(pref_len),
                    "full_len_points": int(len(pts)),
                    "prefix_region_seq_raw": prefix_region_seq_raw,
                    "dest_region_raw": dest_region,
                    "dest_lon": float(lon[-1]),
                    "dest_lat": float(lat[-1]),
                    "dest_x": dest_x,
                    "dest_y": dest_y,
                    "dest_offset_x": offset_x,
                    "dest_offset_y": offset_y,
                    "call_type": row.get("CALL_TYPE"),
                    "origin_call": safe_scalar(row, "ORIGIN_CALL"),
                    "origin_stand": safe_scalar(row, "ORIGIN_STAND"),
                    "taxi_id": safe_scalar(row, "TAXI_ID"),
                    "day_type": row.get("DAY_TYPE"),
                }
            )

            if len(buffers[sp]) >= shard_size:
                _flush_raw_buffer(buffers, raw_shard_root, raw_shard_counts, example_counts, sp)

        if (i + 1) % 100000 == 0:
            total_generated = sum(example_counts.values()) + sum(len(v) for v in buffers.values())
            log(f"Processed {i+1:,}/{n:,} supervised rows; generated {total_generated:,} prefix examples")

    for sp in ["train", "val", "test"]:
        _flush_raw_buffer(buffers, raw_shard_root, raw_shard_counts, example_counts, sp)
        log(f"{sp}: {example_counts[sp]:,} raw prefix examples in {raw_shard_counts[sp]} shard(s)")

    return example_counts, transition_counter, out_counter, in_counter, dest_counter, active_region_ids


def remap_raw_shards_to_final(
    raw_shard_root: Path,
    out_dir: Path,
    tables_dir: Path,
    region_id_map: Dict[int, int],
    shard_size: int,
    max_preview_rows_per_split: int = 10000,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, List[dict]]]:
    """Remap raw region IDs and write final lightweight supervised shards."""
    final_root = out_dir / "supervised_shards"
    _clear_pt_files(final_root)

    final_buffers: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    final_shard_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    final_example_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    preview_dict: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    def flush_final(split: str) -> None:
        if not final_buffers[split]:
            return
        split_dir = final_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_id = final_shard_counts[split]
        path = split_dir / f"{split}_{shard_id:03d}.pt"
        torch.save(final_buffers[split], path)
        final_shard_counts[split] += 1
        final_example_counts[split] += len(final_buffers[split])
        log(f"Saved final {split} shard {shard_id:03d} with {len(final_buffers[split]):,} examples")
        final_buffers[split].clear()

    for sp in ["train", "val", "test"]:
        raw_paths = sorted((raw_shard_root / sp).glob(f"{sp}_raw_*.pt"))
        for raw_path in raw_paths:
            raw_rows = torch.load(raw_path, weights_only=False)
            for ex in raw_rows:
                seq = [region_id_map[r] for r in ex["prefix_region_seq_raw"] if r in region_id_map]
                if not seq or ex["dest_region_raw"] not in region_id_map:
                    continue
                ex2 = dict(ex)
                ex2["prefix_region_seq"] = seq
                ex2["dest_region"] = region_id_map[ex["dest_region_raw"]]
                final_buffers[sp].append(ex2)

                if len(preview_dict[sp]) < max_preview_rows_per_split:
                    preview_dict[sp].append(ex2)

                if len(final_buffers[sp]) >= shard_size:
                    flush_final(sp)
        flush_final(sp)
        log(f"{sp}: {final_example_counts[sp]:,} final examples in {final_shard_counts[sp]} shard(s)")

        preview_rows = []
        for ex in preview_dict[sp]:
            preview_rows.append(
                {
                    "trip_id": ex["trip_id"],
                    "timestamp": ex["timestamp"],
                    "prefix_sample_id": ex["prefix_sample_id"],
                    "prefix_len_points": ex["prefix_len_points"],
                    "full_len_points": ex["full_len_points"],
                    "prefix_region_seq": json.dumps(ex["prefix_region_seq"]),
                    "dest_region": ex["dest_region"],
                    "dest_lon": ex["dest_lon"],
                    "dest_lat": ex["dest_lat"],
                    "dest_offset_x": ex["dest_offset_x"],
                    "dest_offset_y": ex["dest_offset_y"],
                }
            )
        pd.DataFrame(preview_rows).to_csv(tables_dir / f"{sp}_examples_preview.csv", index=False)

    torch.save(preview_dict, out_dir / "supervised_splits_preview.pt")
    return final_example_counts, final_shard_counts, preview_dict


def build_unlabeled_kaggle_test(
    test_path: Path,
    transformer: Transformer,
    bbox: Tuple[float, float, float, float],
    grid: GridSpec,
    chunksize: int,
    min_points: int,
) -> List[dict]:
    if not test_path.exists():
        return []
    out: List[dict] = []
    log("Building unlabeled Kaggle test prefixes...")
    for chunk in iter_csv_chunks(test_path, chunksize):
        for _, row in chunk.iterrows():
            pts = filter_bbox(parse_polyline(row["POLYLINE"]), bbox)
            if len(pts) < min_points:
                continue
            lon = np.array([p[0] for p in pts], dtype=float)
            lat = np.array([p[1] for p in pts], dtype=float)
            x, y = transformer.transform(lon, lat)
            region_raw = xy_to_region_id(np.asarray(x), np.asarray(y), grid)
            prefix_regions = compress_consecutive(region_raw.tolist())
            if not prefix_regions:
                continue
            out.append(
                {
                    "trip_id": row.get("TRIP_ID"),
                    "prefix_len_points": int(len(pts)),
                    "prefix_lonlat": [(float(a), float(b)) for a, b in zip(lon, lat)],
                    "prefix_region_seq_raw": prefix_regions,
                    "last_lon": float(lon[-1]),
                    "last_lat": float(lat[-1]),
                    "call_type": row.get("CALL_TYPE"),
                    "origin_call": safe_scalar(row, "ORIGIN_CALL"),
                    "origin_stand": safe_scalar(row, "ORIGIN_STAND"),
                    "taxi_id": safe_scalar(row, "TAXI_ID"),
                    "day_type": row.get("DAY_TYPE"),
                }
            )
    log(f"Kaggle unlabeled test prefixes: {len(out):,}")
    return out


def sanitize_tag_value(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def tag_values(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return s


def add_semantic_poi_features(poi_features: gpd.GeoDataFrame, tag_cols: Sequence[str]) -> Tuple[gpd.GeoDataFrame, List[str], Dict[str, List[dict]]]:
    """Add semantic multi-hot columns such as amenity_restaurant and shop_supermarket."""
    poi_features = poi_features.copy()
    feature_names: List[str] = []
    top20: Dict[str, List[dict]] = {}

    for c in tag_cols:
        if c not in poi_features.columns:
            poi_features[c] = np.nan
        vals = tag_values(poi_features[c])
        counts = vals.value_counts()
        top20[c] = [{"category": str(k), "count": int(v)} for k, v in counts.head(20).items()]
        log(f"{c} top 20:")
        if len(counts) == 0:
            log("  (none)")
        else:
            for k, v in counts.head(20).items():
                log(f"  {k}: {int(v)}")

        for raw_val in counts.index.tolist():
            col = f"{c}_{sanitize_tag_value(raw_val)}"
            if col in poi_features.columns:
                # If two raw values sanitize to the same column, merge them with OR.
                poi_features[col] = np.maximum(poi_features[col].fillna(0).astype(int), (poi_features[c].astype(str) == str(raw_val)).astype(int))
            else:
                poi_features[col] = (poi_features[c].astype(str) == str(raw_val)).astype(int)
            if col not in feature_names:
                feature_names.append(col)

        flag_col = f"{c}_flag"
        poi_features[flag_col] = poi_features[c].notna().astype(np.int64)
        if flag_col not in feature_names:
            feature_names.append(flag_col)

    return poi_features, feature_names, top20



def safe_cache_label(s: Optional[str]) -> str:
    """Return a filesystem-safe label for OSM cache folders."""
    if s is None or str(s).strip() == "":
        return "current"
    return re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_") or "current"


def build_osm_tables(place: str, regions: gpd.GeoDataFrame, cache_dir: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, List[dict]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    roads_path = cache_dir / "osm_road_edges.pkl"
    pois_path = cache_dir / "osm_pois.pkl"

    if roads_path.exists() and pois_path.exists():
        log("Loading cached OSM roads and POIs...")
        road_edges = pd.read_pickle(roads_path)
        pois = pd.read_pickle(pois_path)
    else:
        log(f"Downloading OSM road network and POIs for {place}...")
        G = ox.graph_from_place(place, network_type="drive")
        _, road_edges = ox.graph_to_gdfs(G)
        tags = {"amenity": True, "shop": True, "tourism": True, "public_transport": True}
        pois = ox.features_from_place(place, tags)
        road_edges.to_pickle(roads_path)
        pois.to_pickle(pois_path)

    road_edges_proj = road_edges.to_crs(regions.crs).reset_index()
    if "geometry" not in road_edges_proj.columns:
        raise ValueError("OSM road edges do not have geometry.")
    road_edges_proj["road_id"] = np.arange(len(road_edges_proj), dtype=np.int64)
    road_edges_proj["length_m"] = road_edges_proj.geometry.length

    log("Joining OSM roads to regions...")
    road_region_join = gpd.sjoin(road_edges_proj, regions[["region_id", "geometry"]], how="inner", predicate="intersects")
    road_region_edges = road_region_join[["road_id", "region_id"]].drop_duplicates().reset_index(drop=True)

    log("Building road-to-road connectivity edges...")
    by_u: Dict[object, List[int]] = defaultdict(list)
    by_v: Dict[object, List[int]] = defaultdict(list)
    for _, r in road_edges_proj[["u", "v", "road_id"]].iterrows():
        by_u[r["u"]].append(int(r["road_id"]))
        by_v[r["v"]].append(int(r["road_id"]))
    rr_src, rr_dst = [], []
    for node, incoming in by_v.items():
        outgoing = by_u.get(node, [])
        for a in incoming:
            for b in outgoing:
                if a != b:
                    rr_src.append(a)
                    rr_dst.append(b)
    road_connect_edges = pd.DataFrame({"src_road_id": rr_src, "dst_road_id": rr_dst}).drop_duplicates().reset_index(drop=True)

    log("Joining OSM POIs to regions...")
    pois_proj = pois.to_crs(regions.crs).copy()

    # OSM POIs are not always stored as points. Important places such as malls,
    # schools, hospitals, parks, universities, and large buildings are often
    # polygons or multipolygons.  For the destination-prediction graph we need a
    # region assignment for each POI, so we convert every valid POI geometry to a
    # safe representative point before the spatial join.  representative_point()
    # is preferred over centroid because it is guaranteed to lie inside the
    # original geometry for polygon-like objects.
    pois_proj = pois_proj[pois_proj.geometry.notna() & (~pois_proj.geometry.is_empty)].copy()
    pois_proj["osm_geom_type"] = pois_proj.geometry.geom_type.astype(str)
    geom_counts = pois_proj["osm_geom_type"].value_counts()
    log("OSM POI geometry types before point conversion:")
    for geom_type, count in geom_counts.items():
        log(f"  {geom_type}: {int(count)}")

    pois_point = pois_proj.copy().reset_index(drop=True)
    pois_point["geometry"] = pois_point.geometry.representative_point()
    pois_point["poi_id"] = np.arange(len(pois_point), dtype=np.int64)

    poi_join = gpd.sjoin(pois_point, regions[["region_id", "geometry"]], how="inner", predicate="within")
    poi_region_edges = poi_join[["poi_id", "region_id"]].drop_duplicates().reset_index(drop=True)
    log(f"POIs joined to active regions after conversion: {poi_join['poi_id'].nunique():,}")

    keep = ["poi_id", "region_id", "geometry", "osm_geom_type"]
    tag_cols = ["amenity", "shop", "tourism", "public_transport"]
    for c in tag_cols + ["name"]:
        if c in poi_join.columns:
            keep.append(c)
    poi_features = poi_join[keep].copy()
    poi_features, poi_feature_names, poi_tag_top20 = add_semantic_poi_features(poi_features, tag_cols)

    return road_edges_proj, poi_features, road_region_edges, road_connect_edges, poi_region_edges, poi_feature_names, poi_tag_top20


def remap_examples(examples: Dict[str, List[dict]], region_id_map: Dict[int, int]) -> Dict[str, List[dict]]:
    remapped: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for sp, rows in examples.items():
        for ex in rows:
            seq = [region_id_map[r] for r in ex["prefix_region_seq_raw"] if r in region_id_map]
            full_seq = [region_id_map[r] for r in ex["full_region_seq_raw"] if r in region_id_map]
            if not seq or not full_seq or ex["dest_region_raw"] not in region_id_map:
                continue
            ex2 = dict(ex)
            ex2["prefix_region_seq"] = seq
            ex2["full_region_seq"] = full_seq
            ex2["dest_region"] = region_id_map[ex["dest_region_raw"]]
            remapped[sp].append(ex2)
    return remapped


def remap_unlabeled(rows: List[dict], region_id_map: Dict[int, int]) -> List[dict]:
    out = []
    for ex in rows:
        seq = [region_id_map[r] for r in ex["prefix_region_seq_raw"] if r in region_id_map]
        if not seq:
            continue
        ex2 = dict(ex)
        ex2["prefix_region_seq"] = seq
        out.append(ex2)
    return out


def tensor_edge(src: Sequence[int], dst: Sequence[int]) -> torch.Tensor:
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)]), dtype=torch.long)


def build_heterodata(
    active_regions: gpd.GeoDataFrame,
    transition_counter: Counter,
    out_counter: Counter,
    in_counter: Counter,
    dest_counter: Counter,
    poi_features: gpd.GeoDataFrame,
    poi_feature_names: List[str],
    poi_region_edges: pd.DataFrame,
    road_edges_proj: gpd.GeoDataFrame,
    road_region_edges: pd.DataFrame,
    road_connect_edges: pd.DataFrame,
) -> Tuple[HeteroData, Dict[str, dict], Dict[str, List[str]]]:
    region_raw_ids = sorted(active_regions["region_id"].astype(int).unique().tolist())
    region_id_map = {rid: i for i, rid in enumerate(region_raw_ids)}
    poi_raw_ids = sorted(poi_features["poi_id"].astype(int).unique().tolist())
    poi_id_map = {pid: i for i, pid in enumerate(poi_raw_ids)}
    road_raw_ids = sorted(road_edges_proj["road_id"].astype(int).unique().tolist())
    road_id_map = {rid: i for i, rid in enumerate(road_raw_ids)}

    data = HeteroData()
    data["region"].num_nodes = len(region_id_map)
    data["poi"].num_nodes = len(poi_id_map)
    data["road"].num_nodes = len(road_id_map)

    # Region features: standardized coordinates, taxi activity, and aggregated POI counts.
    reg = active_regions.copy()
    reg["mapped_id"] = reg["region_id"].map(region_id_map)
    reg = reg.sort_values("mapped_id")
    x = reg["centroid_x"].astype(float).to_numpy()
    y = reg["centroid_y"].astype(float).to_numpy()
    xz = (x - x.mean()) / (x.std() + 1e-8)
    yz = (y - y.mean()) / (y.std() + 1e-8)
    out_arr = np.array([out_counter.get(int(r), 0) for r in reg["region_id"]], dtype=float)
    in_arr = np.array([in_counter.get(int(r), 0) for r in reg["region_id"]], dtype=float)
    dest_arr = np.array([dest_counter.get(int(r), 0) for r in reg["region_id"]], dtype=float)
    activity = np.vstack([np.log1p(out_arr), np.log1p(in_arr), np.log1p(out_arr + in_arr), np.log1p(dest_arr)]).T
    base_region_feature_names = ["centroid_x_z", "centroid_y_z", "log1p_taxi_out", "log1p_taxi_in", "log1p_taxi_total", "log1p_destination_count"]

    if poi_feature_names and not poi_features.empty:
        poi_counts = poi_features.groupby("region_id")[poi_feature_names].sum()
        poi_count_arr = []
        for r in reg["region_id"]:
            if r in poi_counts.index:
                poi_count_arr.append(np.log1p(poi_counts.loc[r].to_numpy(dtype=float)))
            else:
                poi_count_arr.append(np.zeros(len(poi_feature_names), dtype=float))
        poi_count_arr = np.vstack(poi_count_arr)
    else:
        poi_count_arr = np.empty((len(reg), 0), dtype=float)

    region_feat = np.column_stack([xz, yz, activity, poi_count_arr])
    region_feature_names = base_region_feature_names + [f"log1p_poi_count_{name}" for name in poi_feature_names]
    data["region"].x = torch.tensor(region_feat, dtype=torch.float)

    # POI semantic multi-hot features.
    poi = poi_features.copy()
    poi["mapped_id"] = poi["poi_id"].map(poi_id_map)
    poi = poi.drop_duplicates("mapped_id").sort_values("mapped_id")
    if poi_feature_names:
        poi_feat = poi[poi_feature_names].fillna(0).to_numpy(dtype=float)
    else:
        poi_feat = np.empty((len(poi), 0), dtype=float)
    data["poi"].x = torch.tensor(poi_feat, dtype=torch.float)

    # Road features: standardized log length.
    road = road_edges_proj.copy()
    road["mapped_id"] = road["road_id"].map(road_id_map)
    road = road.sort_values("mapped_id")
    length = road["length_m"].fillna(0).to_numpy(dtype=float)
    log_length = np.log1p(length)
    length_z = (log_length - log_length.mean()) / (log_length.std() + 1e-8)
    data["road"].x = torch.tensor(length_z.reshape(-1, 1), dtype=torch.float)
    road_feature_names = ["log1p_length_m_z"]

    rr_src, rr_dst, rr_w = [], [], []
    for (a, b), w in transition_counter.items():
        if a in region_id_map and b in region_id_map:
            rr_src.append(region_id_map[a]); rr_dst.append(region_id_map[b]); rr_w.append(w)
    data["region", "taxi_transition", "region"].edge_index = tensor_edge(rr_src, rr_dst)
    data["region", "taxi_transition", "region"].edge_weight = torch.tensor(rr_w, dtype=torch.float)
    data["region", "rev_taxi_transition", "region"].edge_index = tensor_edge(rr_dst, rr_src)
    data["region", "rev_taxi_transition", "region"].edge_weight = torch.tensor(rr_w, dtype=torch.float)

    pr = poi_region_edges.drop_duplicates()
    pr_src, pr_dst = [], []
    for _, r in pr.iterrows():
        p = int(r["poi_id"]); rg = int(r["region_id"])
        if p in poi_id_map and rg in region_id_map:
            pr_src.append(poi_id_map[p]); pr_dst.append(region_id_map[rg])
    data["poi", "located_in", "region"].edge_index = tensor_edge(pr_src, pr_dst)
    data["region", "has_poi", "poi"].edge_index = tensor_edge(pr_dst, pr_src)

    rre = road_region_edges.drop_duplicates()
    rd_src, rd_dst = [], []
    for _, r in rre.iterrows():
        rd = int(r["road_id"]); rg = int(r["region_id"])
        if rd in road_id_map and rg in region_id_map:
            rd_src.append(road_id_map[rd]); rd_dst.append(region_id_map[rg])
    data["road", "intersects", "region"].edge_index = tensor_edge(rd_src, rd_dst)
    data["region", "has_road", "road"].edge_index = tensor_edge(rd_dst, rd_src)

    rc = road_connect_edges.drop_duplicates()
    c_src, c_dst = [], []
    for _, r in rc.iterrows():
        a = int(r["src_road_id"]); b = int(r["dst_road_id"])
        if a in road_id_map and b in road_id_map:
            c_src.append(road_id_map[a]); c_dst.append(road_id_map[b])
    data["road", "connects_to", "road"].edge_index = tensor_edge(c_src, c_dst)

    mappings = {
        "region_id_map": region_id_map,
        "poi_id_map": poi_id_map,
        "road_id_map": road_id_map,
        "grid": grid_to_dict(active_regions.attrs.get("grid_spec")),
    }
    feature_names = {
        "region_feature_names": region_feature_names,
        "poi_feature_names": poi_feature_names,
        "road_feature_names": road_feature_names,
    }
    return data, mappings, feature_names


def grid_to_dict(grid: Optional[GridSpec]) -> Optional[dict]:
    if grid is None:
        return None
    return {
        "xmin": grid.xmin,
        "ymin": grid.ymin,
        "xmax": grid.xmax,
        "ymax": grid.ymax,
        "cell_size": grid.cell_size,
        "n_cols": grid.n_cols,
        "n_rows": grid.n_rows,
    }


def write_examples_preview(examples: Dict[str, List[dict]], out_dir: Path, max_rows_per_split: int = 10000) -> Dict[str, List[dict]]:
    preview_dict: Dict[str, List[dict]] = {}
    for sp, rows in examples.items():
        preview_rows = rows[:max_rows_per_split]
        preview_dict[sp] = preview_rows
        preview = []
        for ex in preview_rows:
            preview.append(
                {
                    "trip_id": ex["trip_id"],
                    "timestamp": ex["timestamp"],
                    "prefix_sample_id": ex["prefix_sample_id"],
                    "prefix_len_points": ex["prefix_len_points"],
                    "full_len_points": ex["full_len_points"],
                    "prefix_region_seq": json.dumps(ex["prefix_region_seq"]),
                    "full_region_seq": json.dumps(ex.get("full_region_seq", [])),
                    "dest_region": ex["dest_region"],
                    "dest_lon": ex["dest_lon"],
                    "dest_lat": ex["dest_lat"],
                    "dest_offset_x": ex["dest_offset_x"],
                    "dest_offset_y": ex["dest_offset_y"],
                }
            )
        pd.DataFrame(preview).to_csv(out_dir / f"{sp}_examples_preview.csv", index=False)
    return preview_dict


def write_supervised_shards(examples: Dict[str, List[dict]], out_dir: Path, shard_size: int) -> Dict[str, int]:
    shard_root = out_dir / "supervised_shards"
    shard_counts: Dict[str, int] = {}
    for sp, rows in examples.items():
        split_dir = shard_root / sp
        split_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale shards if rerunning into the same output directory.
        for old in split_dir.glob(f"{sp}_*.pt"):
            old.unlink()
        n_shards = 0
        for start in range(0, len(rows), shard_size):
            shard = rows[start:start + shard_size]
            path = split_dir / f"{sp}_{n_shards:03d}.pt"
            torch.save(shard, path)
            n_shards += 1
        shard_counts[sp] = n_shards
        log(f"Saved {n_shards} {sp} shard(s) with {len(rows):,} examples")
    return shard_counts


def make_bundle(out_dir: Path) -> Path:
    bundle_path = out_dir / "porto_data_bundle_trial3.tar.gz"
    include_paths = [
        out_dir / "hetero_graph.pt",
        out_dir / "id_mappings.pkl",
        out_dir / "feature_names.json",
        out_dir / "preprocess_summary.json",
        out_dir / "kaggle_test_prefixes.pt",
        out_dir / "supervised_splits_preview.pt",
        out_dir / "supervised_shards",
    ]
    preview_paths = sorted((out_dir / "tables").glob("*preview.csv"))
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in include_paths:
            if path.exists():
                tar.add(path, arcname=path.relative_to(out_dir))
        for path in preview_paths:
            tar.add(path, arcname=path.relative_to(out_dir))
    log(f"Wrote bundle: {bundle_path}")
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing raw train.csv and test.csv files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/trial3"), help="Directory where processed data and the final tar.gz bundle will be saved.")
    parser.add_argument("--place", type=str, default="Porto, Portugal")
    parser.add_argument(
        "--osm-date",
        type=str,
        default="2014-06-30T23:59:59Z",
        help='Historical OSM snapshot date for Overpass queries, e.g. "2014-06-30T23:59:59Z". Use "current" to query current OSM.',
    )
    parser.add_argument("--cell-size", type=float, default=250.0)
    parser.add_argument("--buffer-m", type=float, default=500.0)
    parser.add_argument("--bbox", type=float, nargs=4, default=[-8.75, 41.05, -8.45, 41.30], help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--num-prefix-samples", type=int, default=5)
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional debug limit. Omit for full data.")
    parser.add_argument("--bounds-sample-rows", type=int, default=None, help="Optional debug limit for bounds scan. Omit for full data.")
    args = parser.parse_args()

    if args.num_prefix_samples <= 0:
        raise ValueError("--num-prefix-samples must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    osm_date = None if args.osm_date is None or str(args.osm_date).lower() == "current" else args.osm_date
    if osm_date is not None:
        ox.settings.overpass_settings = f'[out:json][timeout:180][date:"{osm_date}"]'
        log(f"Using historical OSM snapshot: {osm_date}")
    else:
        log("Using current OSM snapshot")

    cache_dir = args.out_dir / "osm_cache" / safe_cache_label(osm_date)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find {train_path}")

    log(f"Data dir: {args.data_dir}")
    log(f"Output dir: {args.out_dir}")
    log(f"Cell size: {args.cell_size} m")
    log(f"Prefix samples per trip: {args.num_prefix_samples}")
    log(f"Shard size: {args.shard_size}")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    bbox = tuple(args.bbox)  # type: ignore

    bounds_paths = [train_path]
    if test_path.exists():
        bounds_paths.append(test_path)
    bounds = collect_bounds(bounds_paths, transformer, bbox, args.chunksize, args.bounds_sample_rows)
    grid = make_grid(bounds, args.cell_size, args.buffer_m)
    log(f"Grid: {grid.n_rows} rows x {grid.n_cols} cols = {grid.n_rows * grid.n_cols:,} possible regions")

    regions = make_regions_gdf(grid, crs="EPSG:3763")
    regions.attrs["grid_spec"] = grid

    raw_shard_root = args.out_dir / "_raw_supervised_shards"
    raw_example_counts, transition_counter, out_counter, in_counter, dest_counter, active_region_ids = build_examples_and_transition_counts(
        train_path=train_path,
        transformer=transformer,
        bbox=bbox,
        grid=grid,
        regions_lookup=regions[["region_id", "centroid_x", "centroid_y"]],
        chunksize=args.chunksize,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        num_prefix_samples=args.num_prefix_samples,
        min_points=args.min_points,
        seed=args.seed,
        raw_shard_root=raw_shard_root,
        shard_size=args.shard_size,
        max_train_rows=args.max_train_rows,
    )

    active_region_ids = set(active_region_ids) | set(dest_counter.keys()) | set(out_counter.keys()) | set(in_counter.keys())
    active_regions = regions[regions["region_id"].isin(active_region_ids)].copy().reset_index(drop=True)
    active_regions.attrs["grid_spec"] = grid
    log(f"Active regions: {len(active_regions):,}")

    road_edges_proj, poi_features, road_region_edges, road_connect_edges, poi_region_edges, poi_feature_names, poi_tag_top20 = build_osm_tables(args.place, active_regions, cache_dir)

    active_set = set(active_regions["region_id"].astype(int).tolist())
    poi_region_edges = poi_region_edges[poi_region_edges["region_id"].isin(active_set)].copy()
    poi_features = poi_features[poi_features["region_id"].isin(active_set)].copy()
    road_region_edges = road_region_edges[road_region_edges["region_id"].isin(active_set)].copy()

    data, mappings, feature_names = build_heterodata(
        active_regions=active_regions,
        transition_counter=transition_counter,
        out_counter=out_counter,
        in_counter=in_counter,
        dest_counter=dest_counter,
        poi_features=poi_features,
        poi_feature_names=poi_feature_names,
        poi_region_edges=poi_region_edges,
        road_edges_proj=road_edges_proj,
        road_region_edges=road_region_edges,
        road_connect_edges=road_connect_edges,
    )

    final_example_counts, shard_counts, preview_dict = remap_raw_shards_to_final(
        raw_shard_root=raw_shard_root,
        out_dir=args.out_dir,
        tables_dir=tables_dir,
        region_id_map=mappings["region_id_map"],
        shard_size=args.shard_size,
    )
    kaggle_test_raw = build_unlabeled_kaggle_test(test_path, transformer, bbox, grid, args.chunksize, args.min_points)
    kaggle_test = remap_unlabeled(kaggle_test_raw, mappings["region_id_map"])

    log("Saving graph, mappings, feature names, previews, and shards...")
    torch.save(data, args.out_dir / "hetero_graph.pt")
    torch.save(kaggle_test, args.out_dir / "kaggle_test_prefixes.pt")
    with open(args.out_dir / "id_mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)
    with open(args.out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    active_regions.to_pickle(tables_dir / "regions_250m_active.pkl")
    pd.DataFrame([{"src_region_raw": a, "dst_region_raw": b, "weight": w} for (a, b), w in transition_counter.items()]).to_csv(tables_dir / "region_transition_edges_raw.csv", index=False)
    poi_region_edges.to_csv(tables_dir / "poi_region_edges_raw.csv", index=False)
    road_region_edges.to_csv(tables_dir / "road_region_edges_raw.csv", index=False)
    road_connect_edges.to_csv(tables_dir / "road_connect_edges_raw.csv", index=False)
    poi_features.drop(columns="geometry", errors="ignore").to_csv(tables_dir / "poi_features_raw.csv", index=False)
    road_edges_proj.drop(columns="geometry", errors="ignore").to_csv(tables_dir / "road_features_raw.csv", index=False)

    summary = {
        "cell_size_m": args.cell_size,
        "bbox_lonlat": list(bbox),
        "osm_date": osm_date if osm_date is not None else "current",
        "grid": grid_to_dict(grid),
        "num_prefix_samples": args.num_prefix_samples,
        "shard_size": args.shard_size,
        "raw_supervised_example_counts": raw_example_counts,
        "num_region_nodes": int(data["region"].num_nodes),
        "num_poi_nodes": int(data["poi"].num_nodes),
        "num_road_nodes": int(data["road"].num_nodes),
        "num_train_examples": final_example_counts["train"],
        "num_val_examples": final_example_counts["val"],
        "num_test_examples": final_example_counts["test"],
        "num_kaggle_unlabeled_test_prefixes": len(kaggle_test),
        "supervised_shard_counts": shard_counts,
        "edge_types": {str(k): int(data[k].edge_index.shape[1]) for k in data.edge_types},
        "region_feature_dim": int(data["region"].x.shape[1]),
        "poi_feature_dim": int(data["poi"].x.shape[1]),
        "road_feature_dim": int(data["road"].x.shape[1]),
        "poi_tag_top20": poi_tag_top20,
    }
    with open(args.out_dir / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))

    make_bundle(args.out_dir)
    log("Done.")


if __name__ == "__main__":
    main()
