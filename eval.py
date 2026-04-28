"""
eval.py

Shared evaluation utilities used by all five models.

Two entry points:
  compute_metrics   — pure metric computation given top-K predictions.
                      Used directly by the Markov baseline (which already
                      has region IDs) and called internally by evaluate_model.

  evaluate_model    — full eval loop for neural models. Drives a DataLoader,
                      calls model.forward(), extracts top-K, calls compute_metrics.

Expected DataLoader batch format (produced by partner's collate_fn):
  prefix_ids   : LongTensor  [B, T]   padded region ID sequences
  lengths      : LongTensor  [B]      actual (unpadded) sequence lengths
  dest_region  : LongTensor  [B]      true destination region IDs
  dest_lat     : FloatTensor [B]      true destination latitude
  dest_lon     : FloatTensor [B]      true destination longitude
  metadata     : dict of LongTensors [B] with keys:
                   call_type, taxi_id, day_type, hour, dow
"""

from math import radians, cos, sin, asin, sqrt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Core metric computation — works for Markov and neural models alike
# ---------------------------------------------------------------------------

def compute_metrics(
    topk_preds:   torch.Tensor,
    true_regions: torch.Tensor,
    true_lats:    torch.Tensor,
    true_lons:    torch.Tensor,
    centroids:    dict,
    k_values:     tuple = (1, 5),
) -> dict:
    """
    Compute Recall@K and haversine distance for a batch of predictions.

    Args:
        topk_preds:   [B, K] LongTensor — predicted region IDs sorted by
                      confidence descending (col 0 = top-1 prediction)
        true_regions: [B]   LongTensor — ground truth destination region IDs
        true_lats:    [B]   FloatTensor — true destination latitudes
        true_lons:    [B]   FloatTensor — true destination longitudes
        centroids:    dict  {region_id -> (lat, lon)} from cell_centroids.pt
        k_values:     which K values to compute Recall@K for

    Returns:
        dict with keys: recall@1, recall@5, haversine_km (list of per-example errors)
    """
    B = topk_preds.shape[0]
    hits = {k: 0 for k in k_values}
    haversine_errors = []

    for i in range(B):
        true_r   = true_regions[i].item()
        true_lat = true_lats[i].item()
        true_lon = true_lons[i].item()
        preds    = topk_preds[i].tolist()

        for k in k_values:
            if true_r in preds[:k]:
                hits[k] += 1

        # Haversine uses top-1 predicted cell mapped to its centroid
        pred_lat, pred_lon = centroids[preds[0]]
        haversine_errors.append(haversine_km(pred_lat, pred_lon, true_lat, true_lon))

    return {
        **{f'recall@{k}': hits[k] / B for k in k_values},
        'haversine_km': haversine_errors,   # per-example list, aggregated by caller
    }


# ---------------------------------------------------------------------------
# Neural model eval loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    loader,
    centroids: dict,
    device:    torch.device,
    graph_data=None,
    k_values:  tuple = (1, 5),
) -> dict:
    """
    Full evaluation loop for neural models (GRU, GCN, R-GCN, HGT).

    Args:
        model:      nn.Module with forward(prefix_ids, lengths, metadata,
                    region_emb_matrix=None) -> logits [B, num_classes]
        loader:     DataLoader yielding batches in the format described above
        centroids:  {region_id -> (lat, lon)} from cell_centroids.pt
        device:     torch.device
        graph_data: HeteroData on device for graph models; None for pure GRU.
                    The model is responsible for running its GNN encoder and
                    passing the resulting region_emb_matrix to GRUDestinationModel.
        k_values:   which K values to compute Recall@K for

    Returns:
        dict with Recall@1, Recall@5, Mean Haversine (km), Median Haversine (km)
    """
    model.eval()

    all_hits      = {k: 0 for k in k_values}
    all_haversine = []
    total         = 0

    max_k = max(k_values)

    with torch.no_grad():
        for batch in loader:
            prefix_ids  = batch['prefix_ids'].to(device)
            lengths     = batch['lengths'].to(device)
            dest_region = batch['dest_region']          # keep on CPU for centroid lookup
            dest_lat    = batch['dest_lat']
            dest_lon    = batch['dest_lon']
            metadata    = {k: v.to(device) for k, v in batch['metadata'].items()}

            # Forward pass — graph models pass graph_data internally or accept it here
            logits = model(prefix_ids, lengths, metadata, graph_data=graph_data)
            # logits: [B, num_classes]

            # Top-K predicted region IDs
            topk_preds = logits.topk(max_k, dim=-1).indices.cpu()  # [B, max_k]

            metrics = compute_metrics(
                topk_preds, dest_region, dest_lat, dest_lon, centroids, k_values
            )

            B = prefix_ids.shape[0]
            for k in k_values:
                all_hits[k] += int(metrics[f'recall@{k}'] * B)
            all_haversine.extend(metrics['haversine_km'])
            total += B

    return {
        'Recall@1'            : all_hits[1]  / total,
        'Recall@5'            : all_hits[5]  / total,
        'Mean Haversine (km)' : float(np.mean(all_haversine)),
        'Med Haversine (km)'  : float(np.median(all_haversine)),
        'n'                   : total,
    }


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, dict]) -> None:
    """
    Print a formatted comparison table.

    Args:
        results: {model_name -> evaluate_model output dict}

    Example:
        print_results_table({
            'Markov':  markov_results,
            'GRU+HGT': hgt_results,
        })
    """
    header = f'{"Model":<22} {"R@1":>7} {"R@5":>7} {"Mean H":>10} {"Med H":>10}'
    sep    = '=' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        print(
            f'{name:<22} '
            f'{r["Recall@1"]:>7.4f} '
            f'{r["Recall@5"]:>7.4f} '
            f'{r["Mean Haversine (km)"]:>10.3f} '
            f'{r["Med Haversine (km)"]:>10.3f}'
        )
    print(sep)
