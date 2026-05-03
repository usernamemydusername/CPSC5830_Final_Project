#!/usr/bin/env python3
"""
Evaluate a trained heterogeneous GRU model separately on examples whose
true destination region has at least one POI vs no POI.

This is evaluation-only: it does not train.  It is meant for quick error/subgroup
analysis of models such as:
  - full region features + grouped POI nodes
  - base-only region features + grouped POI nodes

Run this from the model directory where train_heterogeneous_group_rgcn_gru.py exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the exact dataset/model/collate logic from the training script.
# This file may live in model/analyses/, while the training script lives in model/.
# Add the parent directory so imports work when running from analyses/.
MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from train_heterogeneous_group_rgcn_gru import (
    ShardedTrajectoryDataset,
    collate_trajectory_batch,
    HeterogeneousGRUModel,
    load_heterogeneous_graph,
    apply_grouped_poi_features,
)
from eval import compute_metrics, print_results_table


def destination_regions_with_poi(graph) -> set[int]:
    """Return compact region IDs that have at least one POI located in them."""
    etype = ("poi", "located_in", "region")
    if etype not in graph.edge_types:
        return set()
    edge_index = graph[etype].edge_index.cpu()
    # edge_index[1] is destination region IDs in poi -> region edges.
    return set(edge_index[1].tolist())


@torch.no_grad()
def evaluate_dest_poi_groups(
    model,
    loader,
    centroids: dict,
    device: torch.device,
    graph_data,
    dest_regions_with_poi: set[int],
    k_values=(1, 5, 10),
) -> Dict[str, dict]:
    model.eval()
    max_k = max(k_values)

    groups = {
        "dest_has_poi": {
            "hits": {k: 0 for k in k_values},
            "haversine": [],
            "loss_sum": 0.0,
            "n": 0,
        },
        "dest_no_poi": {
            "hits": {k: 0 for k in k_values},
            "haversine": [],
            "loss_sum": 0.0,
            "n": 0,
        },
    }

    for batch in loader:
        prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        y = batch["dest_region"].to(device, non_blocking=True)
        metadata = {k: v.to(device, non_blocking=True) for k, v in batch["metadata"].items()}

        logits = model(prefix_ids, lengths, metadata, graph_data=graph_data)
        losses = F.cross_entropy(logits, y, reduction="none").cpu()
        topk_preds = logits.topk(max_k, dim=-1).indices.cpu()

        dest_cpu = batch["dest_region"].cpu()
        mask_has = torch.tensor([int(r.item()) in dest_regions_with_poi for r in dest_cpu], dtype=torch.bool)
        masks = {
            "dest_has_poi": mask_has,
            "dest_no_poi": ~mask_has,
        }

        for group_name, mask in masks.items():
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            metrics = compute_metrics(
                topk_preds=topk_preds[idx],
                true_regions=batch["dest_region"][idx],
                true_lats=batch["dest_lat"][idx],
                true_lons=batch["dest_lon"][idx],
                centroids=centroids,
                k_values=k_values,
            )

            B = int(idx.numel())
            groups[group_name]["n"] += B
            groups[group_name]["loss_sum"] += float(losses[idx].sum().item())
            for k in k_values:
                groups[group_name]["hits"][k] += int(round(metrics[f"recall@{k}"] * B))
            groups[group_name]["haversine"].extend(metrics["haversine_km"])

    out = {}
    total_n = sum(g["n"] for g in groups.values())
    for group_name, g in groups.items():
        n = max(int(g["n"]), 1)
        out[group_name] = {
            "n": int(g["n"]),
            "fraction": float(g["n"] / total_n) if total_n > 0 else 0.0,
            "loss": float(g["loss_sum"] / n),
            **{f"Recall@{k}": float(g["hits"][k] / n) for k in k_values},
            "Mean Haversine (km)": float(np.mean(g["haversine"])) if g["haversine"] else float("nan"),
            "Med Haversine (km)": float(np.median(g["haversine"])) if g["haversine"] else float("nan"),
        }
    return out


def write_csv(path: Path, results: Dict[str, dict]) -> None:
    keys = ["group", "n", "fraction", "loss", "Recall@1", "Recall@5", "Recall@10", "Mean Haversine (km)", "Med Haversine (km)"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for group, metrics in results.items():
            writer.writerow({"group": group, **metrics})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to best_hetero_rgcn_gru.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--edge-set", choices=["full", "taxi_only", "no_poi", "no_road", "urban_context_only", "region_features_only"], default="full")
    p.add_argument("--region-feature-mode", choices=["full", "base"], default="full")
    p.add_argument("--poi-feature-mode", choices=["raw", "grouped"], default="grouped")
    p.add_argument("--poi-group-mapping", type=Path, default=None)
    p.add_argument("--hetero-conv-type", choices=["sage", "graphconv"], default="sage")
    p.add_argument("--region-emb-dim", type=int, default=64)
    p.add_argument("--gnn-hidden", type=int, default=128)
    p.add_argument("--rgcn-layers", type=int, default=2)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-examples", type=int, default=None, help="Optional quick eval cap")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    centroids = torch.load(args.data_dir / "cell_centroids.pt", map_location="cpu", weights_only=False)
    taxi_id_map = torch.load(args.data_dir / "taxi_id_map.pt", map_location="cpu", weights_only=False)
    with open(args.data_dir / "gru_param_config.json") as f:
        gru_config = json.load(f)

    graph, selected_edge_types = load_heterogeneous_graph(args.data_dir, edge_set=args.edge_set)

    if args.region_feature_mode == "base":
        graph["region"].x = graph["region"].x[:, :6].contiguous()
    elif args.region_feature_mode == "full":
        pass
    else:
        raise ValueError(args.region_feature_mode)

    grouped_poi_feature_names = None
    if args.poi_feature_mode == "grouped":
        graph, grouped_poi_feature_names = apply_grouped_poi_features(
            graph, data_dir=args.data_dir, mapping_path=args.poi_group_mapping
        )

    dest_poi_set = destination_regions_with_poi(graph)
    print(f"destination regions with at least one POI: {len(dest_poi_set):,} / {int(graph['region'].num_nodes):,}")
    print("node_feature_dims=", {ntype: int(graph[ntype].x.shape[1]) for ntype in graph.node_types})
    print(f"region_feature_mode={args.region_feature_mode}, poi_feature_mode={args.poi_feature_mode}")
    if grouped_poi_feature_names is not None:
        print(f"grouped_poi_dim={len(grouped_poi_feature_names)}")

    graph = graph.to(device)

    ds = ShardedTrajectoryDataset(
        args.data_dir,
        args.split,
        taxi_id_map,
        seed=123,
        shuffle_shards=False,
        shuffle_within_shard=False,
        max_examples=args.max_examples,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_trajectory_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = HeterogeneousGRUModel(
        graph_data=graph,
        edge_types=selected_edge_types,
        num_regions=int(graph["region"].num_nodes),
        num_taxi_ids=int(gru_config["num_taxi_ids"]),
        region_emb_dim=args.region_emb_dim,
        gnn_hidden=args.gnn_hidden,
        rgcn_layers=args.rgcn_layers,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        use_layer_norm=True,
        conv_type=args.hetero_conv_type,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    results = evaluate_dest_poi_groups(
        model=model,
        loader=loader,
        centroids=centroids,
        device=device,
        graph_data=graph,
        dest_regions_with_poi=dest_poi_set,
        k_values=(1, 5, 10),
    )

    out_json = args.out_dir / f"{args.split}_destination_poi_group_metrics.json"
    out_csv = args.out_dir / f"{args.split}_destination_poi_group_metrics.csv"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    write_csv(out_csv, results)

    print("\nDestination-POI subgroup results:")
    print_results_table({
        "dest_has_poi": results["dest_has_poi"],
        "dest_no_poi": results["dest_no_poi"],
    })
    for group, metrics in results.items():
        print(f"{group}: n={metrics['n']}, frac={metrics['fraction']:.4f}, R@10={metrics['Recall@10']:.4f}, loss={metrics['loss']:.4f}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
