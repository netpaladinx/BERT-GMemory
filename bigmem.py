"""
BERT-Induced Graph Memory
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ... x (1+K) x H x D1 -> ... x (H*D2)
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_tars, n_heads=1, output_value=True):
        super(MultiHeadedAttention, self).__init__()


    def forward(self, source, target):
        """ source: ... x n_heads x n_dims
            target: ... x n_tars x n_heads x n_dims
            (return): attn: ... x n_heads x n_tars
            (return) value: ... x n_heads x (n_heads*n_dims_sm)
        """



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


class BIGMem(nn.Module):
    def __init__(self,
                 index_groups=5, per_group_indices=200, index_dims=64,
                 memory_nodes=100000, memory_dims=768,
                 device=None,

                 dot_product_mode='scaled_dot'):
        """ anchor_dot: 'scaled_dot' or 'unit_key'
        """
        super(BIGMem, self).__init__()
        self.index_groups = index_groups
        self.per_group_indices = per_group_indices
        self.index_dims = index_dims
        self.memory_nodes = memory_nodes
        self.device = device

        self.index_keys = nn.Parameter(torch.empty(self.n_inode_groups, self.n_pgroup_inodes, self.n_inode_key_dims,
                                                   dtype=torch.float32, device=self.device))
        self.index_pointers = nn.Parameter()

        nn.init.normal_(self.index_keys, std=0.01)



        self.n_nodes = n_nodes
        self.n_edges = int(n_nodes ** 2 * density)
        self.n_rels = n_rels
        self.anchor_dot = anchor_dot
        self.anchor_k = anchor_k

        self._node_keys = nn.Parameter(torch.empty(n_nodes, n_dims_sm, dtype=torch.float32, device=device))  # n_nodes x n_indices x n_dims_sm (10.24M)

        self._adjacency_values = nn.Parameter(torch.empty(self.n_edges, dtype=torch.float32, device=device))  # n_edges
        self._node_embeddings = nn.Parameter(torch.empty(n_nodes, n_node_emb_dims, dtype=torch.float32, device=device))

        self.register_buffer('_adjacency_indices',
                             torch.empty(3, self.n_edges, dtype=torch.int64, device=device))  # 3 x n_edges, (rel, node1, node2)
        self.register_buffer('_adjacency',
                             torch.rand(n_rels, n_nodes, n_nodes, dtype=torch.float32, device='cpu'))  # n_rels x n_nodes x n_nodes



        nn.init.normal_(self._keys, std=0.01)
        self.reset_adjacency()

    def forward(self, *args, **kwargs):
        pass

    def anchor(self, elem_reprs, k=None):
        """ elem_reprs: B x n_elems x n_dims
            (return) mapping: B x n_elems x n_nodes
        """
        # B x n_elems x n_dims => B x n_elems x (n_indices*n_dims_sm) => B x n_elems x n_indices x n_dims_sm

        # B x n_elems x n_indices x n_dims_sm, n_nodes x n_indices x n_dims_sm => B x n_elems x n_indices x n_nodes

        # B x n_elems x n_indices x n_nodes => B x n_elems x n_nodes

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
        batch_size, n_elems, _ = mapping.size()
        mapping_2d = mapping.view(batch_size * n_elems, self.n_nodes)  # (B*n_elems) x n_nodes
        losses = []
        for rel, adj in enumerate(self.adjacency):
            attn = torch.select(attn_mat, 1, rel)  # B x n_elems x n_elems
            elem2node = torch.sparse.mm(adj, mapping_2d.t()).t()  #  (B*n_elems) x n_nodes
            node2elem = elem2node.view(batch_size, n_elems, self.n_nodes).transpose(1, 2)  # B x n_nodes x n_elems
            estimated_attn = torch.matmul(mapping, node2elem)  # B x n_elems x n_elems
            loss = torch.sqrt(attn - estimated_attn).sum([1, 2]).mean()
            losses.append(loss)
        return losses

    def read(self, anchor_mapping, elem_contexts):
        """ anchor_mapping: B x n_elems x n_nodes (anchor_k non-zeros per elem)
            elem_contexts: B x n_elems x n_dims
            return: B x n_positions x n_dims

            e.g., 1-hop coverage with constraint read_k:
                elem -> node_i1 -> rel_a, node_j1
                                -> rel_a, node_j2
                                -> rel_b, node_k1
                     -> node_i2 -> rel_b, node_k2
                                -> rel_c, node_l1
                                -> rel_c, node_l2
                     ...        ...
                where #node_i is anchor_k and #node_j, #node_k, #node_l is read_k
        """
        batch_size, n_elems, _ = anchor_mapping.size()
        mapping_val, mapping_ind = torch.topk(anchor_mapping, self.anchor_k, dim=-1, sorted=False)  # B x n_elems x anchor_k (both)
        read_node_embs = self.node_embeddings[mapping_ind.flatten()].reshape(mapping_ind.size() + (-1,))  # B x n_elems x anchor_k x n_dims

        attention_

        hop0_node_embs # B x n_elems x anchor_k x n_dims
        hop0_out <- elem_contexts, hop0_node_embs  # B x n_elems x n_dims_sm

        read_weight <- elem_contexts, hop0_node_embs # B x n_elems x anchor_k

        for _ in range(self.read_hops):
            read_weight = read_weight.reshape(batch_size * n_elems, self.n_nodes)  # (B*n_elems) x n_nodes



        mapping_2d = mapping.view(batch_size * n_elems, self.n_nodes)  # (B*n_elems) x n_nodes
        for rel, adj in enumerate(self.adjacency):
            elem2node = torch.sparse.mm(adj, mapping_2d.t()).t()  # (B*n_elems) x n_nodes
            elem2node = elem2node.view(batch_size, n_elems, self.n_nodes)  # B x n_elems x n_nodes



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

    @property
    def node_embeddings(self):
        return self._node_embeddings

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


class IndexGraph(object):
    """ i_to_m: (list of numpy array (int)) groups x m_nodes,
                i_to_m[g][0] ~ i_to_m[g][m_nodes/i_nodes - 1] for i0,
                i_to_m[g][m_nodes/i_nodes] ~ i_to_m[g][2*m_nodes/i_nodes - 1] for i1,
                ...
        m_to_i: groups x m_nodes
    """
    def __init__(self, groups, i_nodes, m_nodes):
        self.i_to_m = [np.random.permutation(m_nodes) for _ in groups]
        self.m_to_i =
        self.

class MemoryGraph(object):
    pass