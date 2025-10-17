import torch.nn as nn 
import torch
from torch_geometric.nn import GINEConv, LayerNorm, global_mean_pool

class Gnn_classifier(nn.Module):
    def __init__(self, hidden_size, edge_dim, hidden_dim=128):
        # hidden size: feature size of x 
        # edge_dim: feature size of edge
        # hidden dim: latent space feature size
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gnn_conv = GINEConv(self.mlp, edge_dim=edge_dim)
        self.norm = LayerNorm(hidden_dim)
        self.validity_out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _dense_to_pygGNN(self, node, edge):
        # node: (B, N, D)
        # edge: (B, N, N, D_edge)
        
        batch_idx, node_idx = (node.sum(dim=-1) != 0).nonzero(as_tuple=True)  # (N_valid,)
        all_valid_nodes = node[batch_idx, node_idx, :]  # (N_valid, D)
        
        batch_e, i_edge_idx, j_edge_idx = (edge.sum(dim=-1) != 0).nonzero(as_tuple=True)  # (E_valid,)
        all_valid_edges = edge[batch_e, i_edge_idx, j_edge_idx, :]  # (E_valid, D_edge)

        node_map = {}
        for new_idx, (b, n) in enumerate(zip(batch_idx.tolist(), node_idx.tolist())):
            node_map[(b, n)] = new_idx
        
        new_i = [node_map[(b, i)] for b, i in zip(batch_e.tolist(), i_edge_idx.tolist())]
        new_j = [node_map[(b, j)] for b, j in zip(batch_e.tolist(), j_edge_idx.tolist())]
        edge_index = torch.stack([torch.tensor(new_i, device=node.device), torch.tensor(new_j, device=node.device)], dim=0) # (2, E_valid)

        return all_valid_nodes, edge_index, all_valid_edges, batch_idx


    def forward(self, masked_X, masked_E):
        # masked_X: B, N, D if not true last dimension is all zero
        # E: B, N, N, D if not true last dimnesion is all zero
        # node_mask: B, N
        valid_nodes, edge_index, valid_edges, batch_idx = self._dense_to_pygGNN(masked_X, masked_E)
        x = self.gnn_conv(valid_nodes, edge_index, valid_edges)
        x = global_mean_pool(x, batch=batch_idx) # convert to a graphwise mean
        score = self.validity_out_layer(self.norm(x))
        return score