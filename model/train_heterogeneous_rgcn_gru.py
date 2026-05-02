#!/usr/bin/env python3
"""
Train a heterogeneous R-GCN-style encoder + GRU destination prediction model.

This script intentionally does NOT modify:
  - prepare_data4.py
  - gru_encoder.py
  - build_centroids.py
  - eval.py
  - train_homogeneous_gru_baseline.py

Model idea:
  1. Load the existing PyG HeteroData object from hetero_graph.pt.
  2. Run relation-specific message passing over region / poi / road nodes.
  3. Take the learned region node embeddings as the lookup table for prefix_region_seq.
  4. Use the shared GRUDestinationModel for the trajectory encoder, metadata fusion,
     and prediction head.
  5. Train with cross-entropy over compact region IDs and evaluate with eval.py metrics.

Why this is report-friendly:
  - The heterogeneous graph encoder is the only changed component relative to the
    homogeneous GRU baseline; the GRU, metadata embeddings, head, loss, and metrics
    stay aligned for a fair comparison.
  - --edge-set supports ablations needed for the Analysis & Discussion section.
  - The script writes config.json, training_history.csv, best checkpoint, and
    test_metrics.json for each seed.

OOM prevention:
  - Supervised shards are streamed from disk; full splits are never loaded at once.
  - Validation/test metrics are aggregated online; predictions are not stored.
  - num_workers defaults to 0 to avoid each worker loading shards simultaneously.
  - batch size, AMP, gradient accumulation, max example caps, and edge-set ablations
    are configurable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GraphConv, HeteroConv, SAGEConv
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
    """Convert Unix timestamp to UTC hour/day-of-week for reproducible metadata."""
    t = safe_int(ts, 0)
    if t <= 0:
        return 0, 0
    dt = time.gmtime(t)
    return int(dt.tm_hour), int(dt.tm_wday)  # Monday=0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


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
            raise FileNotFoundError(
                f"No shards found for split={split} under {self.data_dir}/supervised_shards/{split}"
            )

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
# Heterogeneous graph utilities
# ---------------------------------------------------------------------------

FULL_EDGE_TYPES = [
    ("region", "taxi_transition", "region"),
    ("region", "rev_taxi_transition", "region"),
    ("poi", "located_in", "region"),
    ("region", "has_poi", "poi"),
    ("road", "intersects", "region"),
    ("region", "has_road", "road"),
    ("road", "connects_to", "road"),
]


def select_edge_types(edge_types: List[tuple], edge_set: str) -> List[tuple]:
    """Return edge types for the main model and ablation variants."""
    existing = set(edge_types)

    def keep(e: tuple) -> bool:
        src, rel, dst = e
        if e not in existing:
            return False
        if edge_set == "full":
            return True
        if edge_set == "taxi_only":
            return src == "region" and dst == "region" and "taxi_transition" in rel
        if edge_set == "no_poi":
            return src != "poi" and dst != "poi"
        if edge_set == "no_road":
            return src != "road" and dst != "road"
        if edge_set == "urban_context_only":
            return not (src == "region" and dst == "region" and "taxi_transition" in rel)
        if edge_set == "region_features_only":
            return False
        raise ValueError(f"Unknown edge_set={edge_set}")

    selected = [e for e in FULL_EDGE_TYPES if keep(e)]
    return selected


def load_heterogeneous_graph(data_dir: Path, edge_set: str) -> tuple[HeteroData, List[tuple]]:
    data = torch.load(data_dir / "hetero_graph.pt", map_location="cpu", weights_only=False)
    selected = select_edge_types(list(data.edge_types), edge_set=edge_set)

    # Remove unused edge types for memory and clearer logging. Node stores stay intact.
    for edge_type in list(data.edge_types):
        if edge_type not in selected:
            del data[edge_type]

    for ntype in data.node_types:
        data[ntype].x = data[ntype].x.float()

    return data, selected


# ---------------------------------------------------------------------------
# Heterogeneous R-GCN-style encoder + shared GRU
# ---------------------------------------------------------------------------

class HeterogeneousRGCNRegionEncoder(nn.Module):
    """
    Relation-specific heterogeneous message passing encoder.

    Implementation detail:
      PyG HeteroConv holds a separate GraphConv module for each relation type.
      Therefore each relation has its own trainable transformation, matching the
      R-GCN idea of relation-specific weight matrices while still supporting
      bipartite relations such as poi -> region and road -> region.
    """

    def __init__(
        self,
        metadata: tuple,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        edge_types: List[tuple],
        use_layer_norm: bool = True,
        conv_type: str = "sage",
    ):
        super().__init__()
        self.node_types, _ = metadata
        self.edge_types = list(edge_types)
        self.dropout = dropout
        self.num_layers = int(num_layers)
        self.use_layer_norm = use_layer_norm

        self.input_lins = nn.ModuleDict({
            ntype: nn.Linear(max(in_dims[ntype], 1), hidden_dim)
            for ntype in self.node_types
        })
        self.empty_feature_params = nn.ParameterDict({
            ntype: nn.Parameter(torch.zeros(1, 1))
            for ntype in self.node_types
            if in_dims[ntype] == 0
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.conv_type = conv_type.lower()
        for layer in range(self.num_layers):
            out_channels = out_dim if layer == self.num_layers - 1 else hidden_dim
            conv_dict = {}
            for edge_type in self.edge_types:
                if self.conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv((-1, -1), out_channels, aggr="mean")
                elif self.conv_type == "graphconv":
                    conv_dict[edge_type] = GraphConv((-1, -1), out_channels, aggr="add")
                else:
                    raise ValueError(f"Unknown conv_type={self.conv_type}")
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            if layer != self.num_layers - 1 and self.use_layer_norm:
                self.norms.append(nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types}))
            else:
                self.norms.append(nn.ModuleDict())

        # Used by the region_features_only ablation or if num_layers=0.
        self.region_out = nn.Linear(hidden_dim, out_dim)

    def _prepared_x_dict(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for ntype in self.node_types:
            x = data[ntype].x
            if x.numel() == 0 or x.shape[1] == 0:
                x = self.empty_feature_params[ntype].expand(data[ntype].num_nodes, 1)
            h = self.input_lins[ntype](x)
            x_dict[ntype] = F.relu(h)
        return x_dict

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._prepared_x_dict(data)
        edge_index_dict = data.edge_index_dict

        if self.num_layers <= 0 or len(self.edge_types) == 0:
            return self.region_out(F.dropout(x_dict["region"], p=self.dropout, training=self.training))

        for layer, conv in enumerate(self.convs):
            h_dict = conv(x_dict, edge_index_dict)

            # Some ablations may leave a node type without incoming messages.
            # Preserve its previous representation so downstream relations still work.
            for ntype in self.node_types:
                if ntype not in h_dict:
                    h_dict[ntype] = x_dict[ntype]

            if layer != len(self.convs) - 1:
                new_dict = {}
                for ntype, h in h_dict.items():
                    if self.use_layer_norm and ntype in self.norms[layer]:
                        h = self.norms[layer][ntype](h)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    new_dict[ntype] = h
                x_dict = new_dict
            else:
                x_dict = h_dict

        return x_dict["region"]


class HeterogeneousGRUModel(nn.Module):
    """Wrapper compatible with eval.py's model(..., graph_data=...) call."""

    def __init__(
        self,
        graph_data: HeteroData,
        edge_types: List[tuple],
        num_regions: int,
        num_taxi_ids: int,
        region_emb_dim: int = 64,
        gnn_hidden: int = 128,
        rgcn_layers: int = 2,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        conv_type: str = "sage",
    ):
        super().__init__()
        in_dims = {ntype: int(graph_data[ntype].x.shape[1]) for ntype in graph_data.node_types}
        self.encoder = HeterogeneousRGCNRegionEncoder(
            metadata=graph_data.metadata(),
            in_dims=in_dims,
            hidden_dim=gnn_hidden,
            out_dim=region_emb_dim,
            num_layers=rgcn_layers,
            dropout=dropout,
            edge_types=edge_types,
            use_layer_norm=use_layer_norm,
            conv_type=conv_type,
        )
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
            raise ValueError("graph_data is required for HeterogeneousGRUModel")
        region_emb = self.encoder(graph_data)
        return self.gru(prefix_ids, lengths, metadata, region_emb_matrix=region_emb)


# ---------------------------------------------------------------------------
# Metrics and training loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    centroids: dict,
    device: torch.device,
    graph_data: HeteroData,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Same metric definitions as eval.py, extended to Recall@10 and online loss aggregation."""
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
            all_hits[k] += int(round(metrics[f"recall@{k}"] * B))
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
    graph_data: HeteroData,
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
    step = 0

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

    if step > 0 and step % grad_accum_steps != 0:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <data-dir>/runs/hetero_rgcn_gru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--region-emb-dim", type=int, default=64)
    parser.add_argument("--gnn-hidden", type=int, default=128)
    parser.add_argument("--rgcn-layers", type=int, default=2)
    parser.add_argument("--gru-hidden", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--edge-set",
        choices=["full", "taxi_only", "no_poi", "no_road", "urban_context_only", "region_features_only"],
        default="full",
        help="Main model uses full. Other options are intended for ablation studies.",
    )
    parser.add_argument("--no-layer-norm", action="store_true", help="Ablation: remove type-wise LayerNorm in hidden R-GCN layers.")
    parser.add_argument(
        "--hetero-conv-type",
        choices=["sage", "graphconv"],
        default="sage",
        help="Relation-specific convolution used inside HeteroConv."
    )
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

    out_dir = args.out_dir or (args.data_dir / "runs" / "hetero_rgcn_gru")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    centroids = torch.load(args.data_dir / "cell_centroids.pt", map_location="cpu", weights_only=False)
    taxi_id_map = torch.load(args.data_dir / "taxi_id_map.pt", map_location="cpu", weights_only=False)
    with open(args.data_dir / "gru_param_config.json") as f:
        gru_config = json.load(f)

    hetero_graph, selected_edge_types = load_heterogeneous_graph(args.data_dir, edge_set=args.edge_set)
    num_regions = int(hetero_graph["region"].num_nodes)
    num_taxi_ids = int(gru_config["num_taxi_ids"])
    node_feature_dims = {ntype: int(hetero_graph[ntype].x.shape[1]) for ntype in hetero_graph.node_types}
    edge_counts = {str(e): int(hetero_graph[e].edge_index.shape[1]) for e in hetero_graph.edge_types}

    print(f"num_regions={num_regions}, num_taxi_ids={num_taxi_ids}")
    print(f"node_feature_dims={node_feature_dims}")
    print(f"edge_set={args.edge_set}")
    print(f"selected_edge_types={selected_edge_types}")
    print(f"edge_counts={edge_counts}")

    hetero_graph = hetero_graph.to(device)

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

    model = HeterogeneousGRUModel(
        graph_data=hetero_graph,
        edge_types=selected_edge_types,
        num_regions=num_regions,
        num_taxi_ids=num_taxi_ids,
        region_emb_dim=args.region_emb_dim,
        gnn_hidden=args.gnn_hidden,
        rgcn_layers=args.rgcn_layers,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        use_layer_norm=(not args.no_layer_norm),
        conv_type=args.hetero_conv_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "num_regions": num_regions,
        "num_taxi_ids": num_taxi_ids,
        "node_feature_dims": node_feature_dims,
        "selected_edge_types": [str(e) for e in selected_edge_types],
        "edge_counts": edge_counts,
        "output_dim_note": "num_dest_classes is set to num_regions so logit indices equal compact region IDs for eval.py.",
        "method_note": "Relation-specific HeteroConv/GraphConv encoder produces region embeddings passed to the shared GRUDestinationModel.",
    })
    save_json(out_dir / "config.json", config)

    history: List[dict] = []
    best_val = -1.0
    best_epoch = 0
    best_path = out_dir / "best_hetero_rgcn_gru.pt"

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            graph_data=hetero_graph,
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
            graph_data=hetero_graph,
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

        # Use Recall@5 as the model-selection metric because destination prediction is top-K.
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

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_with_eval_metrics(
        model=model,
        loader=test_loader,
        centroids=centroids,
        device=device,
        graph_data=hetero_graph,
        k_values=(1, 5, 10),
    )
    save_json(out_dir / "test_metrics.json", test_metrics)

    table_metrics = {
        "HeteroRGCN+GRU": {
            "Recall@1": test_metrics["Recall@1"],
            "Recall@5": test_metrics["Recall@5"],
            "Mean Haversine (km)": test_metrics["Mean Haversine (km)"],
            "Med Haversine (km)": test_metrics["Med Haversine (km)"],
            "n": test_metrics["n"],
        }
    }
    print_results_table(table_metrics)
    print(f"Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
