import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from gru_encoder import GRUDestinationModel


class HGTEncoder(nn.Module):
    """
    2-layer Heterogeneous Graph Transformer (HGT) encoder over the full
    heterogeneous urban graph (region, poi, road nodes + 7 edge types).

    Produces a region embedding matrix [num_regions, hidden_dim] that is
    passed to GRUDestinationModel as region_emb_matrix. The GRU then indexes
    into it using the prefix region ID sequence.

    HGT uses type-specific Q/K/V projections and meta-relation attention,
    parameterized by the triplet <src_type, edge_type, dst_type>. This gives
    it an advantage over R-GCN on sparser edge types (e.g. poi -> region)
    because parameters are shared across relations involving the same node types.

    Args:
        metadata:        output of hetero_graph.metadata() — (node_types, edge_types)
        in_channels_dict: {node_type: feature_dim} — raw feature dim per node type
        hidden_dim:      embedding dimension for all node types (must equal
                         GRUDestinationModel.region_emb_dim, default 64)
        num_heads:       number of attention heads in HGTConv
        num_layers:      number of HGT message passing layers
        dropout:         dropout applied after each HGT layer
    """

    def __init__(
        self,
        metadata:         tuple,
        in_channels_dict: dict,
        hidden_dim:       int = 64,
        num_heads:        int = 4,
        num_layers:       int = 2,
        dropout:          float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout    = nn.Dropout(dropout)

        # Project each node type's raw features into a common hidden_dim space.
        # Uses explicit input dims (passed via in_channels_dict) so parameters
        # are fully initialized before any forward pass.
        node_types = metadata[0]
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_channels_dict[ntype], hidden_dim) for ntype in node_types
        })

        # HGT layers — each layer does type-aware multi-head attention
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: HeteroData on device — the full heterogeneous graph

        Returns:
            region_embs: [num_regions, hidden_dim] — region node embeddings
                         ready to be passed as region_emb_matrix to GRUDestinationModel
        """
        # Project all node types to hidden_dim
        x_dict = {
            ntype: self.dropout(self.input_proj[ntype](data[ntype].x))
            for ntype in data.node_types
        }

        # Run HGT layers with residual connections
        for conv in self.convs:
            x_dict_new = conv(x_dict, data.edge_index_dict)
            # Residual + dropout for each node type
            x_dict = {
                ntype: self.dropout(torch.relu(x_dict_new[ntype])) + x_dict[ntype]
                for ntype in x_dict
            }

        return x_dict['region']   # [num_regions, hidden_dim]


class HGTDestinationModel(nn.Module):
    """
    Full model: HGT graph encoder + GRU trajectory encoder + MLP head.

    At each forward pass:
      1. HGTEncoder runs on the static heterogeneous graph → region embeddings
      2. GRUDestinationModel uses those embeddings to encode the prefix sequence
      3. MLP head predicts the destination region

    The graph is passed in at forward time (not stored in the model) so it can
    live on GPU once and be shared across batches without copying.

    Args:
        metadata:         hetero_graph.metadata()
        in_channels_dict: {node_type: feature_dim} — e.g. {ntype: graph[ntype].x.shape[1]}
        num_regions:      total region nodes — cfg['num_regions']
        num_dest_classes: active destination classes — cfg['num_dest_classes']
        num_taxi_ids:     unique taxi IDs — cfg['num_taxi_ids']
        hidden_dim:       HGT hidden dim AND GRU region embedding dim (must match)
        num_heads:        HGT attention heads
        num_layers:       HGT message passing layers
        gru_hidden:       GRU hidden state size
        gru_layers:       GRU layer count
        dropout:          shared dropout rate across HGT and GRU
    """

    def __init__(
        self,
        metadata:         tuple,
        in_channels_dict: dict,
        num_regions:      int,
        num_dest_classes: int,
        num_taxi_ids:     int,
        hidden_dim:       int = 64,
        num_heads:        int = 4,
        num_layers:       int = 2,
        gru_hidden:       int = 128,
        gru_layers:       int = 2,
        dropout:          float = 0.3,
    ):
        super().__init__()

        self.hgt = HGTEncoder(
            metadata         = metadata,
            in_channels_dict = in_channels_dict,
            hidden_dim       = hidden_dim,
            num_heads        = num_heads,
            num_layers       = num_layers,
            dropout          = dropout,
        )

        self.gru = GRUDestinationModel(
            num_regions      = num_regions,
            num_dest_classes = num_dest_classes,
            num_taxi_ids     = num_taxi_ids,
            region_emb_dim   = hidden_dim,   # must match HGT hidden_dim
            gru_hidden       = gru_hidden,
            gru_layers       = gru_layers,
            dropout          = dropout,
        )

    def forward(
        self,
        prefix_ids:   torch.Tensor,
        lengths:      torch.Tensor,
        metadata:     dict,
        graph_data=None,
        region_embs:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prefix_ids:   [B, T]  padded region ID sequences (long)
            lengths:      [B]     actual sequence lengths (long)
            metadata:     dict of [B] long tensors: call_type, taxi_id,
                          day_type, hour, dow
            graph_data:   HeteroData on device — used to compute region_embs
                          when region_embs is None (e.g. at eval time)
            region_embs:  [num_regions, hidden_dim] pre-computed by the caller
                          (pass this during training to run HGT only once/epoch)

        Returns:
            logits: [B, num_dest_classes]
        """
        if region_embs is None:
            region_embs = self.hgt(graph_data)      # [num_regions, hidden_dim]

        return self.gru(prefix_ids, lengths, metadata, region_emb_matrix=region_embs)
