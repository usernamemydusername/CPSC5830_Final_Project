#!/usr/bin/env python3
"""
Train a homogeneous mobility-GNN + GRU destination prediction baseline.

This script calls:
  - gru_encoder.py
  - build_centroids.py
  - eval.py

Model idea:
  1. Load hetero_graph.pt, but keep only region nodes and taxi_transition edges.
  2. Run a homogeneous GNN over the region mobility graph to produce region embeddings.
  3. Feed prefix_region_seq into the shared GRUDestinationModel using those GNN embeddings.
  4. Train with cross-entropy over compact region IDs.
  5. Track train/validation metrics and write final test metrics.

OOM prevention:
  - Shards are streamed from disk; the full supervised split is never loaded at once.
  - Validation/test loops aggregate metrics online; predictions are not stored.
  - num_workers defaults to 0 to avoid each worker loading shards simultaneously.
  - batch size, max examples, and AMP are configurable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from torch_geometric.nn import GCNConv, SAGEConv
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires torch_geometric. Run it in the same environment used for prepare_data4.py."
    ) from exc

from gru_encoder import GRUDestinationModel
from eval import compute_metrics, print_results_table


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

CALL_TYPE_MAP = {"A": 0, "B": 1, "C": 2}
DAY_TYPE_MAP = {"A": 0, "B": 1, "C": 2}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def timestamp_to_hour_dow(ts) -> tuple[int, int]:
    """Convert Unix timestamp to UTC hour/day-of-week. Porto local shift is not critical for a baseline."""
    t = safe_int(ts, 0)
    if t <= 0:
        return 0, 0
    # Use UTC to avoid extra timezone dependencies on the cluster.
    dt = time.gmtime(t)
    hour = int(dt.tm_hour)
    dow = int(dt.tm_wday)  # Monday=0
    return hour, dow


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sharded dataset and collate function
# ---------------------------------------------------------------------------

class ShardedTrajectoryDataset(IterableDataset):
    """Stream prefix examples from supervised_shards/<split>/*.pt."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        taxi_id_map: Dict,
        seed: int = 123,
        shuffle_shards: bool = False,
        shuffle_within_shard: bool = False,
        max_examples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.taxi_id_map = taxi_id_map
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.max_examples = max_examples
        self.epoch = 0
        self.paths = sorted((self.data_dir / "supervised_shards" / split).glob(f"{split}_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No shards found for split={split} under {self.data_dir}/supervised_shards/{split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _iter_paths_for_worker(self) -> List[Path]:
        paths = list(self.paths)
        rng = random.Random(self.seed + 1009 * self.epoch)
        if self.shuffle_shards:
            rng.shuffle(paths)

        info = get_worker_info()
        if info is not None:
            paths = paths[info.id :: info.num_workers]
        return paths

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed + 9176 * self.epoch)
        yielded = 0
        for path in self._iter_paths_for_worker():
            rows = torch.load(path, map_location="cpu", weights_only=False)
            if self.shuffle_within_shard:
                rng.shuffle(rows)
            for ex in rows:
                yield self._encode_example(ex)
                yielded += 1
                if self.max_examples is not None and yielded >= self.max_examples:
                    return

    def _encode_example(self, ex: dict) -> dict:
        seq = ex.get("prefix_region_seq", [])
        if len(seq) == 0:
            # Should not happen after prepare_data4.py, but keep it safe.
            seq = [0]

        hour, dow = timestamp_to_hour_dow(ex.get("timestamp"))
        raw_taxi = ex.get("taxi_id")
        taxi_idx = self.taxi_id_map.get(raw_taxi, 0)

        return {
            "prefix_ids": torch.tensor(seq, dtype=torch.long),
            "length": int(len(seq)),
            "dest_region": torch.tensor(safe_int(ex.get("dest_region"), 0), dtype=torch.long),
            "dest_lat": torch.tensor(float(ex.get("dest_lat", 0.0)), dtype=torch.float),
            "dest_lon": torch.tensor(float(ex.get("dest_lon", 0.0)), dtype=torch.float),
            "metadata": {
                "call_type": torch.tensor(CALL_TYPE_MAP.get(str(ex.get("call_type", "A")), 0), dtype=torch.long),
                "taxi_id": torch.tensor(int(taxi_idx), dtype=torch.long),
                "day_type": torch.tensor(DAY_TYPE_MAP.get(str(ex.get("day_type", "A")), 0), dtype=torch.long),
                "hour": torch.tensor(hour, dtype=torch.long),
                "dow": torch.tensor(dow, dtype=torch.long),
            },
        }


def collate_trajectory_batch(rows: List[dict]) -> dict:
    prefix_list = [r["prefix_ids"] for r in rows]
    lengths = torch.tensor([r["length"] for r in rows], dtype=torch.long)
    prefix_ids = pad_sequence(prefix_list, batch_first=True, padding_value=0)

    metadata = {}
    for key in ["call_type", "taxi_id", "day_type", "hour", "dow"]:
        metadata[key] = torch.stack([r["metadata"][key] for r in rows], dim=0).long()

    return {
        "prefix_ids": prefix_ids.long(),
        "lengths": lengths,
        "dest_region": torch.stack([r["dest_region"] for r in rows], dim=0).long(),
        "dest_lat": torch.stack([r["dest_lat"] for r in rows], dim=0).float(),
        "dest_lon": torch.stack([r["dest_lon"] for r in rows], dim=0).float(),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Homogeneous GNN + shared GRU
# ---------------------------------------------------------------------------

class HomogeneousRegionEncoder(nn.Module):
    """Two-layer homogeneous GNN over region nodes only."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, gnn_type: str = "sage"):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        if self.gnn_type == "sage":
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
        elif self.gnn_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=True, normalize=True)
        else:
            raise ValueError("--gnn-type must be 'sage' or 'gcn'")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gnn_type == "gcn":
            h = self.conv1(x, edge_index, edge_weight=edge_weight)
        else:
            h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.gnn_type == "gcn":
            h = self.conv2(h, edge_index, edge_weight=edge_weight)
        else:
            h = self.conv2(h, edge_index)
        return h


class HomogeneousGRUModel(nn.Module):
    """Wrapper compatible with eval.py's model(..., graph_data=...) call."""

    def __init__(
        self,
        num_regions: int,
        num_taxi_ids: int,
        region_feat_dim: int,
        region_emb_dim: int = 64,
        gnn_hidden: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
        gnn_type: str = "sage",
    ):
        super().__init__()
        self.encoder = HomogeneousRegionEncoder(
            in_dim=region_feat_dim,
            hidden_dim=gnn_hidden,
            out_dim=region_emb_dim,
            dropout=dropout,
            gnn_type=gnn_type,
        )
        # IMPORTANT: output classes are compact region IDs, because eval.py treats top-k logit
        # indices as predicted region IDs. Therefore use num_regions, not only seen dest classes.
        self.gru = GRUDestinationModel(
            num_regions=num_regions,
            num_dest_classes=num_regions,
            num_taxi_ids=num_taxi_ids,
            region_emb_dim=region_emb_dim,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )

    def forward(self, prefix_ids: torch.Tensor, lengths: torch.Tensor, metadata: dict, graph_data=None) -> torch.Tensor:
        if graph_data is None:
            raise ValueError("graph_data is required for HomogeneousGRUModel")
        region_emb = self.encoder(graph_data.x, graph_data.edge_index, getattr(graph_data, "edge_weight", None))
        return self.gru(prefix_ids, lengths, metadata, region_emb_matrix=region_emb)


@dataclass
class HomoGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "HomoGraph":
        return HomoGraph(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=None if self.edge_weight is None else self.edge_weight.to(device),
        )


def load_homogeneous_region_graph(data_dir: Path, gnn_type: str = "sage") -> HomoGraph:
    hetero = torch.load(data_dir / "hetero_graph.pt", map_location="cpu", weights_only=False)
    x = hetero["region"].x.float()

    edge_index = hetero["region", "taxi_transition", "region"].edge_index.long()
    pieces = [edge_index]
    weights = None
    if ("region", "rev_taxi_transition", "region") in hetero.edge_types:
        rev_edge_index = hetero["region", "rev_taxi_transition", "region"].edge_index.long()
        pieces.append(rev_edge_index)
    edge_index = torch.cat(pieces, dim=1)

    if gnn_type.lower() == "gcn":
        w = hetero["region", "taxi_transition", "region"].edge_weight.float()
        w_pieces = [torch.log1p(w)]
        if ("region", "rev_taxi_transition", "region") in hetero.edge_types:
            rw = hetero["region", "rev_taxi_transition", "region"].edge_weight.float()
            w_pieces.append(torch.log1p(rw))
        weights = torch.cat(w_pieces, dim=0)

    return HomoGraph(x=x, edge_index=edge_index, edge_weight=weights)


# ---------------------------------------------------------------------------
# Metrics and training loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    centroids: dict,
    device: torch.device,
    graph_data: HomoGraph,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Same metric definitions as eval.py, but supports Recall@10 and online aggregation."""
    model.eval()
    max_k = max(k_values)
    all_hits = {k: 0 for k in k_values}
    all_haversine: List[float] = []
    total = 0
    total_loss = 0.0

    for batch in loader:
        prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        y = batch["dest_region"].to(device, non_blocking=True)
        metadata = {k: v.to(device, non_blocking=True) for k, v in batch["metadata"].items()}

        logits = model(prefix_ids, lengths, metadata, graph_data=graph_data)
        loss = F.cross_entropy(logits, y)
        topk_preds = logits.topk(max_k, dim=-1).indices.cpu()

        metrics = compute_metrics(
            topk_preds=topk_preds,
            true_regions=batch["dest_region"],
            true_lats=batch["dest_lat"],
            true_lons=batch["dest_lon"],
            centroids=centroids,
            k_values=k_values,
        )

        B = prefix_ids.shape[0]
        total += B
        total_loss += float(loss.item()) * B
        for k in k_values:
            all_hits[k] += int(metrics[f"recall@{k}"] * B)
        all_haversine.extend(metrics["haversine_km"])

    out = {f"Recall@{k}": all_hits[k] / max(total, 1) for k in k_values}
    out.update({
        "loss": total_loss / max(total, 1),
        "Mean Haversine (km)": float(np.mean(all_haversine)) if all_haversine else float("nan"),
        "Med Haversine (km)": float(np.median(all_haversine)) if all_haversine else float("nan"),
        "n": total,
    })
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    graph_data: HomoGraph,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    track_train_topk: bool = True,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    model.train()
    total = 0
    total_loss = 0.0
    hits = {k: 0 for k in k_values}
    max_k = max(k_values)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        y = batch["dest_region"].to(device, non_blocking=True)
        metadata = {k: v.to(device, non_blocking=True) for k, v in batch["metadata"].items()}

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(prefix_ids, lengths, metadata, graph_data=graph_data)
            loss = F.cross_entropy(logits, y) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        B = y.numel()
        total += B
        total_loss += float(loss.item()) * grad_accum_steps * B

        if track_train_topk:
            with torch.no_grad():
                topk = logits.detach().topk(max_k, dim=-1).indices
                for k in k_values:
                    hits[k] += int((topk[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item())

    # Flush any remaining accumulated gradients.
    if step % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    out = {"loss": total_loss / max(total, 1), "n": total}
    if track_train_topk:
        out.update({f"Recall@{k}": hits[k] / max(total, 1) for k in k_values})
    return out


def write_history_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <data-dir>/runs/homo_gru_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--region-emb-dim", type=int, default=64)
    parser.add_argument("--gnn-hidden", type=int, default=128)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gnn-type", choices=["sage", "gcn"], default="sage")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision to reduce GPU memory.")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Debug/OOM fallback: cap examples per epoch.")
    parser.add_argument("--max-val-examples", type=int, default=None, help="Debug/OOM fallback: cap validation examples.")
    parser.add_argument("--max-test-examples", type=int, default=None, help="Debug/OOM fallback: cap test examples.")
    parser.add_argument("--no-train-topk", action="store_true", help="Only track train loss to reduce per-batch overhead.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = args.out_dir or (args.data_dir / "runs" / "homo_gru_baseline")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load helper files produced by build_centroids.py.
    centroids = torch.load(args.data_dir / "cell_centroids.pt", map_location="cpu", weights_only=False)
    taxi_id_map = torch.load(args.data_dir / "taxi_id_map.pt", map_location="cpu", weights_only=False)
    with open(args.data_dir / "gru_param_config.json") as f:
        gru_config = json.load(f)

    # Load homogeneous region graph from the existing HeteroData.
    homo_graph = load_homogeneous_region_graph(args.data_dir, gnn_type=args.gnn_type).to(device)
    num_regions = int(homo_graph.x.shape[0])
    num_taxi_ids = int(gru_config["num_taxi_ids"])
    region_feat_dim = int(homo_graph.x.shape[1])
    print(f"num_regions={num_regions}, region_feat_dim={region_feat_dim}, num_taxi_ids={num_taxi_ids}")
    print(f"homogeneous edges={homo_graph.edge_index.shape[1]}")

    # Datasets stream shards from disk; they do not load the whole split into memory.
    train_ds = ShardedTrajectoryDataset(
        args.data_dir, "train", taxi_id_map, seed=args.seed,
        shuffle_shards=True, shuffle_within_shard=True, max_examples=args.max_train_examples,
    )
    val_ds = ShardedTrajectoryDataset(
        args.data_dir, "val", taxi_id_map, seed=args.seed,
        shuffle_shards=False, shuffle_within_shard=False, max_examples=args.max_val_examples,
    )
    test_ds = ShardedTrajectoryDataset(
        args.data_dir, "test", taxi_id_map, seed=args.seed,
        shuffle_shards=False, shuffle_within_shard=False, max_examples=args.max_test_examples,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_trajectory_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    model = HomogeneousGRUModel(
        num_regions=num_regions,
        num_taxi_ids=num_taxi_ids,
        region_feat_dim=region_feat_dim,
        region_emb_dim=args.region_emb_dim,
        gnn_hidden=args.gnn_hidden,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "num_regions": num_regions,
        "num_taxi_ids": num_taxi_ids,
        "region_feat_dim": region_feat_dim,
        "output_dim_note": "num_dest_classes is set to num_regions so logit indices equal compact region IDs for eval.py.",
    })
    save_json(out_dir / "config.json", config)

    history: List[dict] = []
    best_val = -1.0
    best_epoch = 0
    best_path = out_dir / "best_homo_gru.pt"

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            graph_data=homo_graph,
            scaler=scaler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            track_train_topk=(not args.no_train_topk),
            k_values=(1, 5, 10),
        )
        val_metrics = evaluate_with_eval_metrics(
            model=model,
            loader=val_loader,
            centroids=centroids,
            device=device,
            graph_data=homo_graph,
            k_values=(1, 5, 10),
        )

        elapsed = time.time() - t0
        row = {"epoch": epoch, "seconds": elapsed}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        write_history_csv(out_dir / "training_history.csv", history)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_R@1={val_metrics['Recall@1']:.4f} | "
            f"val_R@5={val_metrics['Recall@5']:.4f} | "
            f"val_R@10={val_metrics['Recall@10']:.4f} | "
            f"val_meanH={val_metrics['Mean Haversine (km)']:.3f} km | "
            f"{elapsed:.1f}s"
        )

        # Use Recall@5 as the selection metric because destination prediction is naturally top-K.
        score = val_metrics["Recall@5"]
        if score > best_val:
            best_val = score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, best_path)
            print(f"  saved new best checkpoint -> {best_path}")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final test evaluation from the best checkpoint.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_with_eval_metrics(
        model=model,
        loader=test_loader,
        centroids=centroids,
        device=device,
        graph_data=homo_graph,
        k_values=(1, 5, 10),
    )
    save_json(out_dir / "test_metrics.json", test_metrics)

    # Also print a compact table using the existing eval.py printer for R@1/R@5/mean/median.
    table_metrics = {
        "Homogeneous+GRU": {
            "Recall@1": test_metrics["Recall@1"],
            "Recall@5": test_metrics["Recall@5"],
            "Recall@10": test_metrics["Recall@10"],
            "Mean Haversine (km)": test_metrics["Mean Haversine (km)"],
            "Med Haversine (km)": test_metrics["Med Haversine (km)"],
            "n": test_metrics["n"],
        }
    }
    print_results_table(table_metrics)
    # print(f"Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
