import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUDestinationModel(nn.Module):
    """
    Shared trajectory encoder used by all graph models (and the pure GRU baseline).

    The only thing that varies across models is the source of region embeddings:
      - Pure GRU baseline: region_emb_matrix=None  →  uses internal nn.Embedding
      - GCN / R-GCN / HGT: caller passes region_emb_matrix produced by the GNN

    Args:
        num_regions:      total number of region nodes in the graph (6750)
        num_dest_classes: number of active destination classes (2739)
        num_taxi_ids:     number of unique taxi IDs found in training data
        region_emb_dim:   region embedding dimension (must match GNN output dim)
        gru_hidden:       GRU hidden state size
        gru_layers:       number of GRU layers
        dropout:          dropout probability (applied between GRU layers and in MLP)
    """

    def __init__(
        self,
        num_regions: int,
        num_dest_classes: int,
        num_taxi_ids: int,
        region_emb_dim: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.region_emb_dim = region_emb_dim

        # Used only when no external GNN embedding matrix is provided
        self.region_emb = nn.Embedding(num_regions, region_emb_dim)

        # Metadata embeddings
        self.call_type_emb = nn.Embedding(3, 8)       # A=0, B=1, C=2
        self.taxi_id_emb   = nn.Embedding(num_taxi_ids, 16)
        self.day_type_emb  = nn.Embedding(3, 8)       # A=0, B=1, C=2
        self.hour_emb      = nn.Embedding(24, 8)      # 0–23
        self.dow_emb       = nn.Embedding(7, 8)       # 0=Mon … 6=Sun

        meta_dim = 8 + 16 + 8 + 8 + 8  # 48

        self.gru = nn.GRU(
            input_size=region_emb_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden + meta_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_dest_classes),
        )

    def forward(
        self,
        prefix_ids: torch.Tensor,
        lengths: torch.Tensor,
        metadata: dict,
        region_emb_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prefix_ids:         [B, T]  padded region ID sequences (long)
            lengths:            [B]     actual (unpadded) sequence lengths (long)
            metadata:           dict with long tensors of shape [B]:
                                  call_type, taxi_id, day_type, hour, dow
            region_emb_matrix:  [num_regions, region_emb_dim] from a GNN encoder,
                                  or None to use the internal embedding table

        Returns:
            logits: [B, num_dest_classes]
        """
        # --- Region embeddings ---
        if region_emb_matrix is None:
            x = self.region_emb(prefix_ids)        # [B, T, region_emb_dim]
        else:
            x = region_emb_matrix[prefix_ids]      # [B, T, region_emb_dim]

        # --- GRU over prefix ---
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)               # hidden: [gru_layers, B, gru_hidden]
        traj = hidden[-1]                          # [B, gru_hidden]  last layer

        # --- Metadata ---
        meta = torch.cat([
            self.call_type_emb(metadata['call_type']),   # [B, 8]
            self.taxi_id_emb(metadata['taxi_id']),        # [B, 16]
            self.day_type_emb(metadata['day_type']),      # [B, 8]
            self.hour_emb(metadata['hour']),              # [B, 8]
            self.dow_emb(metadata['dow']),                # [B, 8]
        ], dim=-1)                                        # [B, 48]

        # --- Predict ---
        return self.head(torch.cat([traj, meta], dim=-1))  # [B, num_dest_classes]
