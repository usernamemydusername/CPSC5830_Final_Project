"""
build_centroids.py

Run ONCE before any training starts. Scans all training shards and produces
three files saved inside the data bundle folder:

  cell_centroids.pt: compact_region_id -> (lat, lon)
      Used in eval.py to convert a predicted region ID into a GPS coordinate
      so we can compute haversine distance against the true destination.

  taxi_id_map.pt: raw_taxi_id -> compact_idx
      The raw taxi IDs in the shard examples (e.g. 20000380) are not contiguous
      integers, so they cannot be used directly as indices into nn.Embedding.
      This map remaps them to 0-based contiguous indices.
      Used in the training notebook when encoding the taxi_id metadata field,
      and to determine num_taxi_ids when constructing GRUDestinationModel.

  gru_param_config.json
      Exact model constructor parameters derived from the training data.
      Load this in every training notebook instead of hardcoding values.
      Keys: num_regions, num_dest_classes, num_taxi_ids.

Only training shards are scanned — no val/test data is touched to avoid leakage.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import torch
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR        = Path('porto_data_bundle')
CENTROIDS_PATH  = DATA_DIR / 'cell_centroids.pt'
TAXI_MAP_PATH   = DATA_DIR / 'taxi_id_map.pt'
GRU_CONFIG_PATH = DATA_DIR / 'gru_param_config.json'

# ---------------------------------------------------------------------------
# Load grid config and region id mappings
# ---------------------------------------------------------------------------
with open(DATA_DIR / 'preprocess_summary.json') as f:
    summary = json.load(f)

with open(DATA_DIR / 'id_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

grid      = summary['grid']
bbox      = summary['bbox_lonlat']   # [lon_min, lat_min, lon_max, lat_max]
n_cols    = grid['n_cols']           # 100
cell_size = grid['cell_size']        # 250.0 m
xmin, ymin, xmax, ymax = (grid['xmin'], grid['ymin'],
                           grid['xmax'], grid['ymax'])
lon_min, lat_min, lon_max, lat_max = bbox

# Inverse of region_id_map: compact_region_id -> raw flat grid index
# Needed to compute geometric fallback centroids for cells with no training destinations
region_id_map  = mappings['region_id_map']   # raw -> compact
compact_to_raw = {v: k for k, v in region_id_map.items()}


def raw_id_to_latlon(raw_id: int) -> tuple[float, float]:
    """
    Geometric center of a grid cell given its raw flat index.
    Used as a fallback centroid for region cells that never appear as a
    destination in the training set.
    Linear approximation — error < 5m over Porto's small area.
    """
    col      = raw_id % n_cols
    row      = raw_id // n_cols
    x_center = xmin + (col + 0.5) * cell_size
    y_center = ymin + (row + 0.5) * cell_size
    lon = lon_min + (x_center - xmin) / (xmax - xmin) * (lon_max - lon_min)
    lat = lat_min + (y_center - ymin) / (ymax - ymin) * (lat_max - lat_min)
    return lat, lon


# ---------------------------------------------------------------------------
# Single pass over all training shards
# Collect everything we need in one scan to avoid reading 50 shards twice
# ---------------------------------------------------------------------------
shard_paths = sorted((DATA_DIR / 'supervised_shards' / 'train').glob('train_*.pt'))
print(f'Scanning {len(shard_paths)} training shards...')

lat_acc  = defaultdict(list)   # dest_region -> list of dest_lat
lon_acc  = defaultdict(list)   # dest_region -> list of dest_lon
taxi_ids = set()               # all raw taxi_id values seen in training

for i, path in enumerate(shard_paths):
    shard = torch.load(path, map_location='cpu', weights_only=False)
    for ex in shard:
        # For cell_centroids.pt
        cell = ex['dest_region']
        lat_acc[cell].append(ex['dest_lat'])
        lon_acc[cell].append(ex['dest_lon'])

        # For taxi_id_map.pt
        taxi_ids.add(ex['taxi_id'])

    if (i + 1) % 10 == 0:
        print(f'  {i + 1}/{len(shard_paths)} shards done')

print(f'\nUnique destination cells found : {len(lat_acc)}')
print(f'Unique taxi IDs found          : {len(taxi_ids)}')


# ---------------------------------------------------------------------------
# Build cell_centroids.pt
# ---------------------------------------------------------------------------
num_regions     = summary['num_region_nodes']
centroids       = {}
empirical_count = 0
fallback_count  = 0

for compact_id in range(num_regions):
    if compact_id in lat_acc:
        # Empirical mean of all training destinations that fell in this cell.
        # Preferred over geometric center because real drop-off points cluster
        # near entrances, stations, etc. — not at the mathematical cell center.
        lat = float(np.mean(lat_acc[compact_id]))
        lon = float(np.mean(lon_acc[compact_id]))
        empirical_count += 1
    else:
        # Geometric fallback for cells that were never a destination in training.
        # These cells can still be predicted by the model, so we need a coordinate.
        raw_id     = compact_to_raw[compact_id]
        lat, lon   = raw_id_to_latlon(raw_id)
        fallback_count += 1
    centroids[compact_id] = (lat, lon)

torch.save(centroids, CENTROIDS_PATH)
print(f'\ncell_centroids.pt saved -> {CENTROIDS_PATH}')
print(f'  Empirical centroids : {empirical_count}')
print(f'  Geometric fallbacks : {fallback_count}')

# Sanity check: all centroids inside Porto bounding box
lats = [v[0] for v in centroids.values()]
lons = [v[1] for v in centroids.values()]
assert min(lats) >= lat_min - 0.01 and max(lats) <= lat_max + 0.01, 'lat out of bounds'
assert min(lons) >= lon_min - 0.01 and max(lons) <= lon_max + 0.01, 'lon out of bounds'
print('  Bounding box check passed.')


# ---------------------------------------------------------------------------
# Build taxi_id_map.pt
# Map raw taxi IDs (e.g. 20000380) to 0-based contiguous indices so they
# can be used directly as indices into nn.Embedding(num_taxi_ids, 16).
# The number of entries (len(taxi_id_map)) is passed as num_taxi_ids when
# constructing GRUDestinationModel.
# ---------------------------------------------------------------------------
taxi_id_map = {raw_id: idx for idx, raw_id in enumerate(sorted(taxi_ids))}

torch.save(taxi_id_map, TAXI_MAP_PATH)
print(f'\ntaxi_id_map.pt saved -> {TAXI_MAP_PATH}')
print(f'  num_taxi_ids = {len(taxi_id_map)}')
print(f'  Sample mapping: {list(taxi_id_map.items())[:5]}')


# ---------------------------------------------------------------------------
# Save gru_param_config.json
# Stores the exact values needed to construct GRUDestinationModel so that
# training notebooks never need to hardcode or recompute these numbers.
#
#   num_regions      — total region nodes in the graph (from preprocess_summary.json)
#   num_dest_classes — unique destination regions found across ALL train shards
#                      (definitive count; only computable after full scan)
#   num_taxi_ids     — unique taxi IDs found across ALL train shards
# ---------------------------------------------------------------------------
gru_config = {
    'num_regions'      : int(summary['num_region_nodes']),
    'num_dest_classes' : int(summary['num_region_nodes']),  # predict over all regions
    'num_taxi_ids'     : len(taxi_id_map),
}

with open(GRU_CONFIG_PATH, 'w') as f:
    json.dump(gru_config, f, indent=2)

print(f'\ngru_param_config.json saved -> {GRU_CONFIG_PATH}')
print(f'  {gru_config}')
