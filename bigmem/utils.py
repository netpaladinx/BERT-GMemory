import math

from numba import njit, prange
import numpy as np

import torch
import torch.nn as nn
import torch_sparse


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def scaled_dot_softmax(a, b):
    """
    Args:
         a (Tensor): ... x dims
         b (Tensor): ... x n x dims
         "..." means same or able to broadcast
    Rets:
        sd_softmax (Tensor): ... x n
    """
    scaled_dot = b.matmul(a.unsqueeze(-1)).squeeze(-1).div(math.sqrt(a.size(-1)))  # ... x n
    sd_softmax = scaled_dot.softmax(-1)  # ... x n
    return sd_softmax


def to_sparse(val, ind, row_max, col_max):
    """
    Args:
         val (Tensor): ... x k
         ind (LongTensor): ... x k

    Rets:
        (Tensor, LongTensor): N, 2 x N
    """
    row_0 = torch.arange(row_max).unsqueeze(-1).repeat(1, ind.size(-1)).reshape(-1).to(ind)
    row_1 = ind.reshape(-1)
    ind = torch.stack([row_0, row_1], 0)  # 2 x ...
    val = val.reshape(-1)
    ind, val = torch_sparse.coalesce(ind, val, row_max, col_max)
    return val, ind


def transpose(val, ind, m, n):
    ind, val = torch_sparse.transpose(ind, val, m, n)
    return val, ind


def spspmm(val1, ind1, val2, ind2, m, k, n):
    ind, val = torch_sparse.spspmm(ind1, val1, ind2, val2, m, k, n)
    return val, ind


def batch_spspmm(val1, ind1, val2, ind2, b, m, k, n):
    ind1 = ind1.clone()
    ind2 = ind2.clone()
    ind1[1] = ind1[1] + ind1[0] / m * k
    ind2[0] = ind2[0] + ind2[1] / n * k
    ind, val = torch_sparse.spspmm(ind1, val1, ind2, val2, b * m, b * k, b * n)
    ind[1] = ind[1] % n
    return val, ind


def mse(a, b):
    return torch.mean(torch.sum((a - b)**2, (1,2)))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def to_tensor(arr, device=None):
    return torch.from_numpy(arr).to(device=device)


@njit(parallel=True)
def group_topk(seg, val, k, n):
    res = np.empty(n * k, dtype=np.int64)
    val = -val
    for i in prange(len(seg) - 1):
        arr = val[seg[i]:seg[i+1]]
        res[i*k : (i+1)*k] = np.argsort(arr)[:k] + seg[i]
    return res

def sparse_topk(val, ind, k, n):
    ind_np = to_numpy(ind[0])
    val_np = to_numpy(val)
    seg = np.concatenate((np.unique(ind_np, return_index=True)[1], [len(ind_np)]))
    res = group_topk(seg, val_np, k, n)
    res = to_tensor(res, device=ind.device)
    return val[res], ind[:, res]


def center(a):
    return a - a.mean(-1, keepdim=True)


@njit
def take_first_k_new(candidates, new_elems, k, old_elems):  # old_elems
    candidates = np.sort(candidates)
    old_elems = np.sort(old_elems)
    len1 = len(candidates)
    len2 = len(old_elems)
    i, p1, p2 = 0, 0, 0
    while i < k and p1 < len1 and p2 < len2 :
        if candidates[p1] == old_elems[p2]:
            p1 += 1
            p2 += 1
        elif candidates[p1] < old_elems[p2]:
            new_elems[i] = candidates[p1]
            i += 1
            p1 += 1
        else:
            p2 += 1

    while i < k and p1 < len1:
        new_elems[i] = candidates[p1]
        i += 1
        p1 += 1


@njit(parallel=True)
def sample_new_edges(new_edges, num_neighbors, old_edges=None):
    n_nodes = len(new_edges)
    for i in prange(len(new_edges)):
        samples = np.random.choice(n_nodes, num_neighbors, replace=False)

        if old_edges is None:
            new_edges[i] = samples
        else:
            old_elems = old_edges[i]
            take_first_k_new(samples, new_edges[i], num_neighbors - len(old_elems), old_elems)


@njit(parallel=True)
def sample_multi_graphs(new_graphs, num_neighbors, old_graphs=None):
    for i in prange(len(new_graphs)):
        if old_graphs is None:
            sample_new_edges(new_graphs[i], num_neighbors)
        else:
            sample_new_edges(new_graphs[i], num_neighbors, old_edges=old_graphs[i])
