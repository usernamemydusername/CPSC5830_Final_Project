"""
build_gru_config.py

Scans all training shards and writes gru_param_config.json — the exact
constructor parameters needed to build GRUDestinationModel (and any GNN
model that wraps it).

Run this once before training if gru_param_config.json doesn't exist yet.
build_centroids.py also writes this file as a side-effect, so you only
need this script if you want the config without recomputing centroids.

Output — porto_data_bundle/gru_param_config.json:
    num_regions      : total region nodes (from preprocess_summary.json)
    num_dest_classes : unique destination regions seen in training shards
    num_taxi_ids     : unique taxi IDs seen in training shards
"""

from pathlib import Path
import json
import torch

DATA_DIR        = Path('porto_data_bundle')
GRU_CONFIG_PATH = DATA_DIR / 'gru_param_config.json'

with open(DATA_DIR / 'preprocess_summary.json') as f:
    summary = json.load(f)

shard_paths = sorted((DATA_DIR / 'supervised_shards' / 'train').glob('train_*.pt'))
print(f'Scanning {len(shard_paths)} training shards...')

dest_regions = set()
taxi_ids     = set()

for i, path in enumerate(shard_paths):
    shard = torch.load(path, map_location='cpu', weights_only=False)
    for ex in shard:
        dest_regions.add(ex['dest_region'])
        taxi_ids.add(ex['taxi_id'])
    if (i + 1) % 10 == 0:
        print(f'  {i + 1}/{len(shard_paths)} shards done')

config = {
    'num_regions'      : int(summary['num_region_nodes']),
    'num_dest_classes' : len(dest_regions),
    'num_taxi_ids'     : len(taxi_ids),
}

with open(GRU_CONFIG_PATH, 'w') as f:
    json.dump(config, f, indent=2)

print(f'\ngru_param_config.json saved -> {GRU_CONFIG_PATH}')
print(f'  {config}')