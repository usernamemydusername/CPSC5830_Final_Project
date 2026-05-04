import torch
import torch.nn as nn


class MLPDestinationModel(nn.Module):
    """
    Baseline 2 — MLP on Region Features.

    No sequence modeling, no graph encoder. Encodes a trajectory prefix
    using only the first and last region's precomputed feature vectors
    (from data['region'].x), concatenated with metadata embeddings.

    This ablation tests how much signal lives in "where the trip started +
    where it currently is" vs the full sequence (GRU) or urban graph (GNN).

    Args:
        region_feat_dim:  dimension of data['region'].x — 150
        num_dest_classes: number of destination region classes
        num_taxi_ids:     number of unique taxi IDs
        hidden_dim:       MLP hidden layer size
        dropout:          dropout probability
    """

    def __init__(
        self,
        region_feat_dim:  int,
        num_dest_classes: int,
        num_taxi_ids:     int,
        hidden_dim:       int = 256,
        dropout:          float = 0.3,
    ):
        super().__init__()

        # Metadata embeddings — identical to GRU model for fair comparison
        self.call_type_emb = nn.Embedding(3, 8)
        self.taxi_id_emb   = nn.Embedding(num_taxi_ids, 16)
        self.day_type_emb  = nn.Embedding(3, 8)
        self.hour_emb      = nn.Embedding(24, 8)
        self.dow_emb       = nn.Embedding(7, 8)

        meta_dim  = 8 + 16 + 8 + 8 + 8   # 48
        input_dim = region_feat_dim * 2 + meta_dim  # first + last region + metadata

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dest_classes),
        )

    def forward(
        self,
        first_ids:    torch.Tensor,
        last_ids:     torch.Tensor,
        metadata:     dict,
        region_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            first_ids:    [B] first region ID of each prefix (long)
            last_ids:     [B] last region ID of each prefix (long)
            metadata:     dict of [B] long tensors: call_type, taxi_id,
                          day_type, hour, dow
            region_feats: [num_regions, region_feat_dim] — data['region'].x

        Returns:
            logits: [B, num_dest_classes]
        """
        first_feat = region_feats[first_ids]   # [B, region_feat_dim]
        last_feat  = region_feats[last_ids]    # [B, region_feat_dim]

        # Metadata embeddings
        meta = torch.cat([
            self.call_type_emb(metadata['call_type']),
            self.taxi_id_emb(metadata['taxi_id']),
            self.day_type_emb(metadata['day_type']),
            self.hour_emb(metadata['hour']),
            self.dow_emb(metadata['dow']),
        ], dim=-1)                             # [B, 48]

        x = torch.cat([first_feat, last_feat, meta], dim=-1)  # [B, input_dim]
        return self.mlp(x)                                     # [B, num_dest_classes]
