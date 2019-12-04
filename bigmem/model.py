"""
BERT-Induced Graph Memory
"""
import numpy as np
import torch
import torch.nn as nn

from bigmem.utils import Lambda, gelu, scaled_dot_softmax, to_sparse, transpose, spspmm, mse, \
    sparse_topk, center, to_numpy, to_tensor, sample_multi_graphs


class Indexer(nn.Module):
    def __init__(self, input_dims, index_dims, index_sizes, k):
        super(Indexer, self).__init__()
        self.dims = index_dims
        self.depth = len(index_sizes)
        self.k = k

        self.query_fn = nn.Linear(input_dims, self.depth * index_dims)

        keys = []
        for i, sz in enumerate(index_sizes):
            if i == 0:
                keys.append(nn.Parameter(torch.empty(sz, index_dims)))
            else:
                keys.append(nn.Parameter(torch.empty(index_sizes[i-1], sz // index_sizes[i-1], index_dims)))
        self.keys =nn.ParameterList(keys)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for key in self.keys:
            nn.init.normal_(key, std=0.01)

    def forward(self, input_hiddens):
        """
        Args:
             input_hiddens (Tensor): ... x input_dims
        """
        queries = self.query_fn(input_hiddens).reshape(input_hiddens.size()[:-1] + (self.depth, self.dims))  # ... x depth x dims

        for i in range(self.depth):
            query = queries.select(-2, i)  # ... x dims
            key = self.keys[i]  # width_0 x dims or width_{i-1} x (width_i/width_{i-1}) x dims

            if i == 0:
                attn = scaled_dot_softmax(query, key)  # ... x width_0
                attn_topk_val, attn_topk_ind = attn.topk(self.k, sorted=False)  # ... x k
                attn_topk_val = attn_topk_val / attn_topk_val.sum(-1, keepdim=True)
            else:
                query = query.unsqueeze(-2)  # ... x 1 x dims
                key = key[attn_topk_ind]  # ... x k x (width_i/width_{i-1}) x dims

                attn = attn_topk_val.unsqueeze(-1) * scaled_dot_softmax(query, key)  # ... x k x (width_i/width_{i-1})
                attn = attn.reshape(attn.size()[:-2] + (-1,))  # ... x (k*width_i/width_{i-1})
                attn_topk_val, attn_topk_ind = attn.topk(self.k, sorted=False)  # ... x k
                attn_topk_val = attn_topk_val / attn_topk_val.sum(-1, keepdim=True)

        return attn_topk_val, attn_topk_ind


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dims, mem_dims, heads, intermediate_dims):
        super(MultiHeadAttention, self).__init__()
        self.dims = mem_dims
        self.heads = heads

        self.query_fn = nn.Linear(input_dims, heads * mem_dims)
        self.output_fn = nn.Sequential(nn.Linear(heads * mem_dims, intermediate_dims),
                                       Lambda(gelu),
                                       nn.Linear(intermediate_dims, input_dims),
                                       nn.LayerNorm(input_dims))

    def forward(self, input_hiddens, keys, values):
        """
        Args:
             input_hiddens (Tensor): ... x dims
             keys (Tensor): ... x heads x slots x dims
             values (Tensor): ... x heads x slots x dims
        """
        queries = self.query_fn(input_hiddens).reshape(input_hiddens.size()[:-1] + (self.heads, self.dims))  # ... x heads x dims

        attn = scaled_dot_softmax(queries, values)  # ... x heads x slots
        attd_values = torch.matmul(attn.unsqueeze(-2), values).squeeze(-2)  # .... x heads x dims
        attd_values = attd_values.reshape(attd_values.size()[:-2] + (-1,))  # ... x (heads*dims)
        attd_values = self.output_fn(attd_values)  # ... x input_dims
        return attd_values


class BIGMem(nn.Module):
    def __init__(self, hidden_dims=768, index_sizes=(100, 1000, 10000, 100000), index_dims=64, index_k=10,
                 num_relations=12, read_k=100, mem_dims=64, wiring_k=100, intermediate_dims=3072):
        super(BIGMem, self).__init__()
        self.num_nodes = index_sizes[-1]
        self.read_k = read_k
        self.wiring_k = wiring_k
        self.num_relations = num_relations

        self.memory_keys = nn.Parameter(torch.empty(num_relations, self.num_nodes, mem_dims))
        self.memory_values = nn.Parameter(torch.empty(num_relations, self.num_nodes, mem_dims))

        self.indexer = Indexer(hidden_dims, index_dims, index_sizes, index_k)
        self.mh_attention = MultiHeadAttention(hidden_dims, mem_dims, num_relations, intermediate_dims)

        self.weight_std = 1.
        self.wiring_weights = nn.Parameter(torch.empty(num_relations, self.num_nodes, wiring_k))
        self.wiring_edges_np = np.empty((num_relations, self.num_nodes, wiring_k), dtype=np.int64)
        self.register_buffer('wiring_edges', to_tensor(self.wiring_edges_np))

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(self.memory_keys, std=0.01)
        nn.init.normal_(self.memory_values, std=0.01)

        self.rewire()

    def anchor(self, elem_hiddens):
        """Anchors input elements onto top-k responsive memory nodes

        Args:
            elem_hiddens (Tensor): batch_size x num_elems x hidden_dims

        Rets:
            elems_to_nodes (Tensor, LongTensor): batch_size x num_elems x index_k
        """
        elems_to_nodes = self.indexer(elem_hiddens)
        return elems_to_nodes

    def write(self, elems_to_elems, elems_to_nodes):
        """ Writes relations between elements into the graph memory

        Args:
            elems_to_elems (Tensor): batch_size x heads x num_elems x num_elems
            elems_to_nodes (tuple (Tensor, LongTensor)): batch_size x num_elems x index_k

        Rets:
            write_loss (scalar)
        """
        batch_size = elems_to_elems.size(0)
        num_elems = elems_to_elems.size(2)

        # elements -> nodes
        e2n_val, e2n_ind = elems_to_nodes
        e2n_val, e2n_ind = to_sparse(e2n_val, e2n_ind, batch_size * num_elems, self.num_nodes)

        # nodes -> elements
        wide_e2n_ind = torch.stack([e2n_ind[0], e2n_ind[1] + e2n_ind[0] / num_elems * self.num_nodes], 0)
        n2e_val, wide_n2e_ind = transpose(e2n_val, wide_e2n_ind, self.num_nodes, batch_size * num_elems)

        write_loss = 0
        elems_toto_nodes = []
        for r in range(self.num_relations):
            # nodes -> nodes
            n2n_val, n2n_ind = self.wiring_weights[r], self.wiring_edges[r]
            n2n_val, n2n_ind = to_sparse(n2n_val, n2n_ind, self.num_nodes, self.num_nodes)

            # elements -> nodes -> nodes
            e22n_val, e22n_ind = spspmm(e2n_val, e2n_ind, n2n_val, n2n_ind,
                                        batch_size * num_elems, self.num_nodes, self.num_nodes)
            elems_toto_nodes.append((e22n_val, e22n_ind))

            # elements -> nodes -> nodes -> elements
            wide_e22n_ind = torch.stack([e22n_ind[0], e22n_ind[1] + e22n_ind[0] / num_elems * self.num_nodes], 0)
            e222e_val, wide_e222e_ind = spspmm(e22n_val, wide_e22n_ind, n2e_val, wide_n2e_ind,
                                          batch_size * num_elems, batch_size * self.num_nodes, batch_size * num_elems)
            e222e_ind = torch.stack([wide_e222e_ind[0], wide_e222e_ind[1] % num_elems], 0)
            e222e = torch.sparse_coo_tensor(e222e_ind, e222e_val, size=(batch_size * num_elems, num_elems))
            e222e = e222e.to_dense().reshape(batch_size, num_elems, num_elems)

            # elements -> elements
            e2e = elems_to_elems.select(1, r)

            write_loss += mse(e222e, e2e)

        return write_loss, elems_toto_nodes

    def read(self, elem_hiddens, elems_to_nodes, elems_toto_nodes):
        """ Read memory values based on mapped nodes and their one-hop neighbors

        Args:
            elem_hiddens (Tensor): batch_size x num_elems x hidden_dims
            elems_to_nodes (tuple (Tensor, LongTensor)): batch_size x num_elems x index_k
            elems_toto_nodes (list of tuple (Tensor, LongTensor)): nnz, 2 x nnz

        Rets:
            mem_read (Tensor): batch_size x num_elems x hidden_dims
        """
        batch_size = elems_to_nodes[0].size(0)
        num_elems = elems_to_nodes[0].size(1)

        # e2n_val, e2n_ind: batch_size x num_elems x index_k
        _, e2n_ind = elems_to_nodes

        mem_keys = []
        mem_vals = []

        for r in range(self.num_relations):
            mem_keys.append(self.memory_keys[r][e2n_ind])  # batch_size x num_elems x index_k x mem_dims
            mem_vals.append(self.memory_values[r][e2n_ind])  # batch_size x num_elems x index_k x mem_dims

            e22n_val, e22n_ind = elems_toto_nodes[r]
            # e22n_ind: 2 x nnz (nnz=batch_size*num_elems*read_k)
            _, e22n_ind = sparse_topk(e22n_val, e22n_ind, self.read_k, batch_size * num_elems)
            # e22n_ind: batch_size x num_elems x read_k
            e22n_ind = e22n_ind[1].reshape(batch_size, num_elems, self.read_k)

            mem_keys.append(self.memory_keys[r][e22n_ind])  # batch_size x num_elems x read_k x mem_dims
            mem_vals.append(self.memory_values[r][e22n_ind])  # batch_size x num_elems x read_k x mem_dims

        mem_keys = torch.cat(mem_keys, 2)  # batch_size x num_elems x (num_relations*(index_k+read_k)) x mem_dims
        mem_keys = mem_keys.reshape(mem_keys.size()[:2] + (self.num_relations, -1, mem_keys.size(-1)))
        mem_vals = torch.cat(mem_vals, 2)  # batch_size x num_elems x num_relations x (index_k+read_k) x mem_dims
        mem_vals = mem_vals.reshape(mem_vals.size()[:2] + (self.num_relations, -1, mem_vals.size(-1)))

        mem_read = self.mh_attention(elem_hiddens, mem_keys, mem_vals)  # batch_size x num_elems x hidden_dims
        return mem_read

    def rewire(self, percent=None):
            if percent is None:
                sample_multi_graphs(self.wiring_edges_np, self.wiring_k)
                self.wiring_edges.copy_(to_tensor(self.wiring_edges_np, device=self.wiring_edges.device))
                nn.init.normal_(self.wiring_weights, std=self.weight_std)
            else:
                kept_k = self.wiring_k - int(self.wiring_k * percent)

                with torch.no_grad():
                    wiring_w = center(self.wiring_weights)
                    self.weight_std = wiring_w.std(-1).mean()
                    topk_val, topk_ind = wiring_w.topk(kept_k, dim=-1, sorted=False)

                    self.wiring_weights[:, :, :kept_k] = wiring_w.gather(-1, topk_ind)
                    self.wiring_edges_np[:, :, :kept_k] = np.take_along_axis(self.wiring_edges_np,
                                                                             to_numpy(topk_ind), -1)

                    sample_multi_graphs(self.wiring_edges_np[:, :, kept_k:], self.wiring_k,
                                        old_graphs=self.wiring_edges_np[:, :, :kept_k])
                    self.wiring_edges.copy_(to_tensor(self.wiring_edges_np, device=self.wiring_edges.device))
                    self.wiring_weights[:, :, kept_k:] = torch.normal(0., self.weight_std,
                                                                      (self.num_relations, self.num_nodes,
                                                                       self.wiring_k - kept_k)).to(self.wiring_weights)


class BIGMemBertLayer(nn.Module):
    def __init__(self, bert_layer):
        super(BIGMemBertLayer, self).__init__()
        self.bert_layer = bert_layer
        self.bigmem = BIGMem()

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        outputs = self.bert_layer(hidden_states)
        assert len(outputs) == 2, "`config.output_attentions` is required to be True"
        attn_probs = outputs[1]  # batch_size x heads x num_elems x num_elems

        elem_hiddens, elems_to_elems = hidden_states, attn_probs
        elems_to_nodes = self.bigmem.anchor(elem_hiddens)
        write_loss, elems_toto_nodes = self.bigmem.write(elems_to_elems, elems_to_nodes)
        mem_read = self.bigmem.read(elem_hiddens, elems_to_nodes, elems_toto_nodes)
        outputs = (mem_read,) + outputs[1:] + (write_loss,)
        return outputs


def convert_bert_to_bert_bigmem(bert_model, layer_idx):
    bert_encoder = bert_model.encoder
    bert_layer = bert_encoder.layer[layer_idx]
    bert_layer_with_bigmem = BIGMemBertLayer(bert_layer)
    bert_encoder.layer[layer_idx] = bert_layer_with_bigmem
    return bert_model
