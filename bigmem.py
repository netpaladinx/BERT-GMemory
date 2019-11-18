"""
BERT-Induced Graph Memory
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_op1(indices1, values1, indices2, values2, N2):
    """ indices1: (LongTensor) D1 x D2 x D3 x K1 (K1 out of N1)
        values1: (FloatTensor) D1 x D2 x D3 x K1 (K1 out of N1)
        indices2: (LongTensor) D3 x N1 x K2 (K2 out of N2)
        values2: (FloatTensor) D3 x N1 x K2 (K2 out of N2)
        (return) out: (tensor) D1 x D2 x K2 (K2 out of N1*N2)
    """
    indices = []
    values = []
    for ind1, val1, ind2, val2 in zip(indices1.unbind(dim=2), values1.unbind(dim=2),  # D1 x D2 x K1
                                      indices2.unbind(dim=0), values2.unbind(dim=0)):  # N1 x K2
        V1 = val1.reshape(-1, 1)  # (D1*D2*K1) x 1
        V2 = val2.index_select(0, ind1.reshape(-1))  # (D1*D2*K1) x K2
        V = torch.reshape(V1 * V2, -1)  # (D1*D2*K1*K2)
        values.append(V)

        I_D1 = torch.arange(ind1.size(0)).reshape(-1, 1).repeat(1, V.numel() // ind1.size(0)).reshape(-1)  # (D1*D2*K1*K2)
        I_D2 = torch.arange(ind1.size(1)).reshape(1, -1, 1).repeat(ind1.size(0), 1, ind1.size(2) * ind2.size(-1)).reshape(-1)  # (D1*D2*K1*K2)
        I_K2 = ind2.index_select(0, ind1.reshape(-1)).reshape(ind1.size() + (ind2.size(-1),))  # D1 x D2 x K1 x K2
        I = torch.stack([I_D1, I_D2, I_K2], dim=0)  # 3 x D1 x D2 x K1 x K2
        indices.append(I.reshape(3, -1))  # 3 x (D1*D2*K1*k2)

    indices = torch.cat(indices, dim=1)  # 3 x (D3*D1*D2*K1*K2)
    values = torch.cat(values, dim=0)  # (D3*D1*D2*K1*K2)
    out = torch.sparse_coo_tensor(indices, values, size=(indices1.size(0), indices1.size(1), N2)).coalesce()  # (sparse) D1 x D2 x N2





class BIGMem(nn.Module):
    def __init__(self, hidden_dims=768,
                 index_groups=5, per_group_indices=200, index_dims=64,
                 memory_nodes=100000, memory_dims=768,
                 epsilon=1e08):
        super(BIGMem, self).__init__()
        self.hidden_dims = hidden_dims
        self.index_groups = index_groups
        self.per_group_indices = per_group_indices
        self.index_dims = index_dims
        self.memory_nodes = memory_nodes
        self.memory_dims = memory_dims
        self.epsilon = epsilon

        self.index_graph = IndexGraph()
        self.memory_graph = MemoryGraph()

        self.index_keys = nn.Parameter(torch.empty(self.index_groups, self.per_group_indices, self.index_dims))
        self.index_graph_weights = nn.Parameter(torch.empty(self.index_groups, self.per_group_indices,
                                                            self.memory_nodes // self.per_group_indices))

        self.memory_embeddings =
        self.memory_graph_weights

        self.fn_elem_queries = nn.Linear(self.hidden_dims, self.index_groups * self.index_dims)

        nn.init.normal_(self.index_keys, std=0.01)

        with torch.no_grad():
            self.index_graph_weights.copy_(torch.from_numpy(self.index_graph.weights).to(self.index_graph_weights))

        self.memory_from_index = []
        self.memory_from_memory = []

    def initialize_memory_from_index(self):
        """ memory_from_index: (list of sparse tensor) (memory_from_index[g]: memory_nodes x per_group_indices)
        """
        weights = self.index_graph_weights / self.index_graph_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        weights = weights.reshape(self.index_groups, self.memory_nodes)
        edges_m_from_i = torch.from_numpy(self.index_graph.edges_m_from_i).to(device=weights.device)
        self.memory_from_index = [torch.sparse_coo_tensor(edges_m_from_i[g], weights[g],
                                                          size=(self.memory_nodes, self.per_group_indices))
                                  for g in range(self.index_groups)]

    def initialize_memory_from_memory(self):




    def forward(self, *args, **kwargs):
        pass

    def anchor(self, elem_hiddens):
        """ elem_hiddens: B x n_elems x hidden_dims
            (return) elem_to_memory: B x n_elems x index_groups x per_group_indices
        """
        # elem_hiddens => elem_queries (B x n_elems x index_groups x index_dims)
        elem_queries = self.fn_elem_queries(elem_hiddens)  # B x n_elems x (index_groups*index_dims)
        elem_queries = torch.reshape(elem_queries, (elem_queries.size(0), elem_queries.size(1),
                                                    self.index_groups, self.index_dims))

        # elem_queries, index_keys (index_groups x per_group_indices x index_dims)
        #   => elem_to_index (B x n_elems x index_groups x per_group_indices)
        elem_to_index = torch.matmul(elem_queries.unsqueeze(3), self.index_keys.transpose(1, 2)).squeeze(3)
        elem_to_index = elem_to_index / math.sqrt(self.index_dims)
        elem_to_index = torch.softmax(elem_to_index, 3)

        # elem_to_index, memory_from_index (memory_nodes x per_group_indices) => memory_from_elem (memory_nodes x n_elems)
        elem_to_memory = [torch.sparse.mm(self.memory_from_index[g], )
                          for g in range(self.index_groups)]

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
    def __init__(self, index_groups, per_group_indices, memory_nodes, beta=1.0):
        self.index_groups = index_groups
        self.per_group_indices = per_group_indices
        self.memory_nodes = memory_nodes
        self.beta = beta

        edges = [np.random.permutation(self.memory_nodes) for _ in range(self.index_groups)]
        edges = np.stack(edges, axis=0)
        self.edges = np.reshape(edges, (self.index_groups, self.per_group_indices,
                                        self.memory_nodes // self.per_group_indices))

        self.weights = np.random.exponential(self.beta, self.edges.shape)

    @property
    def edges_m_from_i(self):
        # (return) (tensor) index_groups x 2 (for pair (m, i)) x memory_nodes
        i = np.reshape(np.arange(self.per_group_indices), (1, self.per_group_indices, 1))
        i = np.tile(i, (self.index_groups, 1, self.memory_nodes // self.per_group_indices))
        i = np.reshape(i, (self.index_groups, self.memory_nodes))
        m = np.reshape(self.edges, (self.index_groups, self.memory_nodes))
        m_from_i = np.stack([m, i], axis=1)  # inidex_groups x 2 x memory_nodes
        return m_from_i


class MemoryGraph(object):
    pass