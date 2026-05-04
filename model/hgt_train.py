"""
hgt_train.py

Standalone training script for GRU + HGT destination prediction model.
Equivalent to hgt.ipynb but runs without Jupyter — submit via run_hgt_train.sh.

Usage:
    python hgt_train.py --data-dir porto_data_bundle --out-dir hgt_results --seeds 0 1 2
"""

import argparse
import datetime
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval import evaluate_model, print_results_table
from hgt_model import HGTDestinationModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardDataset(Dataset):
    def __init__(self, shard_paths, desc='Loading'):
        self.examples = []
        for p in tqdm(shard_paths, desc=desc, unit='shard'):
            self.examples.extend(
                torch.load(p, map_location='cpu', weights_only=False)
            )

    def __len__(self):  return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


CALL_TYPE_MAP = {'A': 0, 'B': 1, 'C': 2}
DAY_TYPE_MAP  = {'A': 0, 'B': 1, 'C': 2}


def collate_fn(batch, taxi_id_map):
    seqs    = [torch.tensor(ex['prefix_region_seq'], dtype=torch.long) for ex in batch]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    prefix_ids = pad_sequence(seqs, batch_first=True, padding_value=0)

    dest_region = torch.tensor([ex['dest_region'] for ex in batch], dtype=torch.long)
    dest_lat    = torch.tensor([ex['dest_lat']    for ex in batch], dtype=torch.float)
    dest_lon    = torch.tensor([ex['dest_lon']    for ex in batch], dtype=torch.float)

    hours, dows = [], []
    for ex in batch:
        dt = datetime.datetime.fromtimestamp(ex['timestamp'], datetime.timezone.utc)
        hours.append(dt.hour)
        dows.append(dt.weekday())

    return {
        'prefix_ids' : prefix_ids,
        'lengths'    : lengths,
        'dest_region': dest_region,
        'dest_lat'   : dest_lat,
        'dest_lon'   : dest_lon,
        'metadata': {
            'call_type': torch.tensor([CALL_TYPE_MAP[ex['call_type']] for ex in batch], dtype=torch.long),
            'taxi_id'  : torch.tensor([taxi_id_map.get(ex['taxi_id'], 0) for ex in batch], dtype=torch.long),
            'day_type' : torch.tensor([DAY_TYPE_MAP[ex['day_type']]   for ex in batch], dtype=torch.long),
            'hour'     : torch.tensor(hours, dtype=torch.long),
            'dow'      : torch.tensor(dows,  dtype=torch.long),
        }
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(seed, cfg, graph, centroids, train_paths, val_loader, test_loader,
               batch_size, device, out_dir, taxi_id_map):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'\n{"="*60}')
    print(f'SEED {seed}')
    print(f'{"="*60}')

    in_channels_dict = {ntype: graph[ntype].x.shape[1] for ntype in graph.node_types}

    model = HGTDestinationModel(
        metadata         = graph.metadata(),
        in_channels_dict = in_channels_dict,
        num_regions      = cfg['num_regions'],
        num_dest_classes = cfg['num_dest_classes'],
        num_taxi_ids     = cfg['num_taxi_ids'],
        hidden_dim       = 64,
        num_heads        = 4,
        num_layers       = 2,
        gru_hidden       = 128,
        gru_layers       = 2,
        dropout          = 0.2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    gru_optimizer = torch.optim.Adam(model.gru.parameters(), lr=1e-3)
    hgt_optimizer = torch.optim.Adam(model.hgt.parameters(), lr=1e-4)
    criterion     = nn.CrossEntropyLoss()
    ckpt_path     = out_dir / f'hgt_best_seed{seed}.pt'

    _collate      = partial(collate_fn, taxi_id_map=taxi_id_map)
    train_dataset = ShardDataset(train_paths, desc=f'Loading train shards (seed {seed})')
    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=4, pin_memory=True,
    )
    print(f'Train: {len(train_dataset):,} examples  |  {len(train_loader):,} batches/epoch\n')

    best_recall5     = 0.0
    patience_counter = 0
    PATIENCE         = 5
    MAX_EPOCHS       = 20
    n_batches        = len(train_loader)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        epoch_start = time.time()

        pbar = tqdm(train_loader, total=n_batches,
                    desc=f'Epoch {epoch:02d}/{MAX_EPOCHS}', unit='batch', leave=True)
        for batch in pbar:
            prefix_ids = batch['prefix_ids'].to(device)
            lengths    = batch['lengths']
            dest       = batch['dest_region'].to(device)
            meta       = {k: v.to(device) for k, v in batch['metadata'].items()}

            region_embs = model.hgt(graph)
            logits      = model(prefix_ids, lengths, meta, region_embs=region_embs)
            loss        = criterion(logits, dest)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            gru_optimizer.step();  gru_optimizer.zero_grad()
            hgt_optimizer.step();  hgt_optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss   = total_loss / n_batches
        epoch_time = time.time() - epoch_start

        print(f'  Epoch {epoch:02d} | train loss {avg_loss:.4f} | '
              f'time {epoch_time/60:.1f} min | validating...', end=' ', flush=True)

        val_results = evaluate_model(model, val_loader, centroids, device, graph_data=graph)
        recall5     = val_results['Recall@5']

        print(f'val R@1 {val_results["Recall@1"]:.4f} | '
              f'val R@5 {recall5:.4f} | '
              f'val R@10 {val_results["Recall@10"]:.4f} | '
              f'mean H {val_results["Mean Haversine (km)"]:.3f} km', end='')

        if recall5 > best_recall5:
            best_recall5 = recall5
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
            print('  ← best')
        else:
            patience_counter += 1
            print(f'  (patience {patience_counter}/{PATIENCE})')
            if patience_counter >= PATIENCE:
                print(f'  Early stopping at epoch {epoch}.')
                break

    print(f'\nLoading best checkpoint (val R@5 = {best_recall5:.4f}) and evaluating on test...')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    test_results = evaluate_model(model, test_loader, centroids, device, graph_data=graph)
    print(f'[Seed {seed}] Test R@1 {test_results["Recall@1"]:.4f} | '
          f'R@5 {test_results["Recall@5"]:.4f} | '
          f'R@10 {test_results["Recall@10"]:.4f} | '
          f'Mean H {test_results["Mean Haversine (km)"]:.3f} km | '
          f'Med H {test_results["Med Haversine (km)"]:.3f} km')
    return test_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   type=Path, default=Path('porto_data_bundle'))
    parser.add_argument('--out-dir',    type=Path, default=Path('hgt_results'))
    parser.add_argument('--batch-size', type=int,  default=512)
    parser.add_argument('--seeds',      type=int,  nargs='+', default=[0, 1])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.data_dir / 'gru_param_config.json') as f:
        cfg = json.load(f)
    print('Config:', cfg)

    taxi_id_map = torch.load(args.data_dir / 'taxi_id_map.pt',    weights_only=False)
    centroids   = torch.load(args.data_dir / 'cell_centroids.pt', weights_only=False)
    graph       = torch.load(args.data_dir / 'hetero_graph.pt',   map_location=device, weights_only=False)
    print(f'Taxi IDs: {len(taxi_id_map)}  |  Centroids: {len(centroids)}')
    print(graph)

    _collate = partial(collate_fn, taxi_id_map=taxi_id_map)

    train_paths = sorted((args.data_dir / 'supervised_shards' / 'train').glob('train_*.pt'))
    val_paths   = sorted((args.data_dir / 'supervised_shards' / 'val').glob('val_*.pt'))
    test_paths  = sorted((args.data_dir / 'supervised_shards' / 'test').glob('test_*.pt'))

    print('Loading val and test shards...')
    val_dataset  = ShardDataset(val_paths,  desc='Loading val shards')
    test_dataset = ShardDataset(test_paths, desc='Loading test shards')

    val_loader  = DataLoader(val_dataset,  batch_size=args.batch_size, shuffle=False,
                             collate_fn=_collate, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=_collate, num_workers=4, pin_memory=True)
    print(f'Val: {len(val_dataset):,}  |  Test: {len(test_dataset):,}')

    seed_results = []
    for seed in args.seeds:
        result = train_seed(seed, cfg, graph, centroids, train_paths,
                            val_loader, test_loader, args.batch_size,
                            device, args.out_dir, taxi_id_map)
        seed_results.append(result)

    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Mean Haversine (km)', 'Med Haversine (km)']
    print('\n=== GRU + HGT — seed summary ===')
    for m in metrics:
        vals = [r[m] for r in seed_results]
        print(f'  {m:<25}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    hgt_avg = {m: float(np.mean([r[m] for r in seed_results])) for m in metrics}
    hgt_avg['n'] = seed_results[0]['n']
    print_results_table({'GRU + HGT (ours)': hgt_avg})

    results_path = args.out_dir / 'hgt_results.json'
    with open(results_path, 'w') as f:
        json.dump({'seed_results': seed_results, 'avg': hgt_avg}, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
