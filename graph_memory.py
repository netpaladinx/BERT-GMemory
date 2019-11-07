import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_mask(input, k, dim=-1):
    _, indices = torch.topk(input, k, dim=dim, sorted=False)
    mask = torch.zeros_like(input).scatter_(dim, indices, 1.)
    return mask


def masked_softmax(input, mask, dim=-1):
    input = input - torch.max(input, dim, keepdim=True)[0]
    exp = torch.exp(input) * mask
    sum = torch.sum(exp, dim, keepdim=True)
    softmax = exp / sum
    return softmax


def split_into_sparse_tensors(indices, values, sparse_size, dim=0):
    sp_tensors = []
    rest_dims = [d for d in range(indices.size(0)) if d != dim]
    for i in range(sparse_size[dim]):
        col_mask = (indices[dim] == i)
        ind = indices[rest_dims][:, col_mask]
        val = values[col_mask]
        sp = torch.sparse_coo_tensor(ind, val, size=list(sparse_size) + list(values.size())[1:]).coalesce()
        sp_tensors.append(sp)
    return sp_tensors


class GraphMemory(nn.Module):
    def __init__(self, n_anchor_dims=256, n_content_dims=768, n_nodes=10000, n_rels=12, density=0.01, device=None,
                 anchor_dot='scaled_dot', anchor_k=8):
        """ anchor_dot: 'scaled_dot' or 'unit_key'
        """
        super(GraphMemory, self).__init__()
        self.n_nodes = n_nodes
        self.n_edges = int(n_nodes ** 2 * density)
        self.n_rels = n_rels
        self.anchor_dot = anchor_dot
        self.anchor_k = anchor_k

        self.register_parameter('_keys',
                                nn.Parameter(torch.empty(n_nodes, n_anchor_dims, dtype=torch.float32, device=device)))  # n_nodes x n_anchor_dims
        self.register_parameter('_adjacency_values',
                                nn.Parameter(torch.empty(self.n_edges, dtype=torch.float32, device=device)))  # n_edges
        self.register_buffer('_adjacency_indices',
                             torch.empty(3, self.n_edges, dtype=torch.int64, device=device))  # 3 x n_edges, (rel, node1, node2)
        self.register_buffer('_adjacency',
                             torch.rand(n_rels, n_nodes, n_nodes, dtype=torch.float32, device='cpu'))  # n_rels x n_nodes x n_nodes

        nn.init.normal_(self._keys, std=0.01)
        self.reset_adjacency()

    def forward(self, *args, **kwargs):
        pass

    def anchor(self, elem_queries, k=None):
        """ elem_queries: B x n_elems x n_anchor_dims
            (return) mapping: B x n_elems x n_nodes
        """
        mapping = torch.matmul(elem_queries, self.keys.t())  # B x n_elems x n_nodes
        k = k or self.anchor_k
        mask = topk_mask(mapping.detach(), k)  # B x n_elems x n_nodes
        mapping = masked_softmax(mapping, mask)  # B x n_elems x n_nodes
        return mapping

    def write(self, mapping, attn_mat):
        """ mapping: B x n_elems x n_nodes
            attn_mat: B x n_heads x n_elems (elem1) x n_elems (elem2)
            return: losses
        """
        batch_size = mapping.size(0)
        n_elems = mapping.size(1)
        mapping_2d = mapping.view(batch_size * n_elems, self.n_nodes)  # (B*n_elems) x n_nodes
        losses = []
        for rel, adj in enumerate(self.adjacency):
            attn = torch.select(attn_mat, 1, rel)  # B x n_elems x n_elems
            out = torch.sparse.mm(adj, mapping_2d.t()).t()  #  (B*n_elems) x n_nodes
            out = out.view(batch_size, n_elems, self.n_nodes).transpose(1, 2)  # B x n_nodes x n_elems
            estimated_attn = torch.matmul(mapping, out)  # B x n_elems x n_elems
            loss = torch.sqrt(attn - estimated_attn).sum([1, 2]).mean()
            losses.append(loss)
        return losses

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
        if self.anchor_dot == 'scaled_dot':
            n_dims = self._keys.size(-1)
            keys = self._keys / math.sqrt(n_dims)
            return keys
        elif self.anchor_dot == 'unit_key':
            keys = F.normalize(self._keys, dim=-1)
            return keys
        else:
            raise ValueError

    @property
    def adjacency(self):
        """ (return) adjacency: list of sparse tensors
                    adjacency[rel_idx]: (node_idx1, node_idx2) -> 1, sparse size: n_nodes x n_nodes
        """
        adjacency_values = torch.clamp(self._adjacency_values, 0, 1)
        return split_into_sparse_tensors(self._adjacency_indices, adjacency_values, [self.n_nodes, self.n_nodes])

    def update_adjacency(self):
        self._adjacency[self._adjacency_indices.unbind()] = self._adjacency_values.detach().cpu()
        self.reset_adjacency()

    def reset_adjacency(self):
        values, indices = torch.topk(self._adjacency.view(-1), self.n_edges, sorted=False)
        indices, ind = torch.sort(indices)
        values = values[ind]
        self._adjacency_indices[0] = indices / (self.n_nodes * self.n_nodes)
        rest = indices % (self.n_nodes * self.n_nodes)
        self._adjacency_indices[1] = rest / self.n_nodes
        self._adjacency_indices[2] = rest % self.n_nodes
        self._adjacency_values.data.copy_(values)
