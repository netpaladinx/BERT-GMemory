import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparse(input, k):
    values, indices = torch.topk(input, k, dim=-1)
    indices = torch.unique(indices)



class GraphMemory(nn.Module):
    def __init__(self, n_dims=768, n_nodes=10000, n_rels=12, density=0.01, device=None):
        super(GraphMemory, self).__init__()
        self.n_nodes = n_nodes
        self._keys = nn.Parameter(torch.empty(n_nodes, n_dims, device=device))  # n_nodes x n_dims
        self._adjacency = nn.Parameter(torch.empty(n_rels, n_nodes, n_nodes, device='cpu'))  # n_rels x n_nodes x n_nodes

        nn.init.normal_(self._keys, std=0.01)
        nn.init.zeros_(self._adjacency)

    def forward(self, mode='anchor'):
        pass

    def anchor(self, positions, k):
        """ positions: B x n_positions x n_dims
            return: B x n_positions x n_nodes, normalized along dim=-1 (sum up to 1)
        """

        mapping = torch.tensordot(positions, self.keys.t(), dims=1)  # B x n_positions x n_nodes
        mapping = topk_sparse(mapping)  # sparse tensor, (idx, position, node) -> 1

    def write(self, anchor_mapping, attention_matrices):
        """ anchor_mapping: B x n_positions x n_nodes, normalized along dim=-1 (sum up to 1)
            attention_matrices: list of size n_heads, attention_matrices[i]: B x n_positions x n_positions x n_heads (n_heads = n_rels)
            return: loss
        """
        pass

    def read(self, anchor_mapping, position_contexts):
        """ anchor_mapping: B x n_positions x n_nodes, normalized along dim=-1 (sum up to 1)
            position_contexts: B x n_positions x n_dims
            return: B x n_positions x n_dims

            position -> node: a(position_context, node)
            position -> node -> rel, nei_node: a(position_context, node) * a(position_context, rel, nei_node)
        """
        pass

    @property
    def keys(self):
        return F.normalize(self._keys, dim=-1)

    @property
    def adjacency(self):
        """ return: sparse tensor, (node_idx1, node_idx2, rel_idx) -> 1, sparse size: n_nodes x n_nodes x n_rels
                n_edges = n_nodes x n_nodes x density
                n_candidate_edges = n_nodes x n_nodes x n_rels
        """
        pass
