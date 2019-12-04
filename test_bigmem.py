import numpy as np
import torch
from bigmem import BIGMem
from bigmem.utils import sample_multi_graphs


def test_bigmem():
    batch_size = 8
    num_elems = 16
    hidden_dims = 32
    index_sizes = (100, 1000, 10000)
    index_dims = 8
    heads = 4
    num_relations = 4
    mem_dims = 8
    intermediate_dims = 512

    graph_memory = BIGMem(hidden_dims=hidden_dims, index_sizes=index_sizes, index_dims=index_dims,
                          num_relations=num_relations, mem_dims=mem_dims, intermediate_dims=intermediate_dims).cuda()

    # test `anchor()`
    elem_hiddens = torch.randn(batch_size, num_elems, hidden_dims).cuda()
    elems_to_nodes = graph_memory.anchor(elem_hiddens)

    # test `write()`
    elems_to_elems = torch.rand(batch_size, heads, num_elems, num_elems).cuda()
    write_loss, elems_toto_nodes = graph_memory.write(elems_to_elems, elems_to_nodes)

    # test `read()`
    mem_read = graph_memory.read(elem_hiddens, elems_to_nodes, elems_toto_nodes)

    # test `rewire()`
    graph_memory.rewire(percent=0.5)


def test_rewire():
    num_relations = 2
    num_nodes = 100
    k = 10
    edges = np.empty((num_relations, num_nodes, k), dtype=np.int64)
    sample_multi_graphs(edges, k)
    sample_multi_graphs(edges[:, :, 5:], k, old_graphs=edges[:, :, :5])

if __name__ == '__main__':
    test_bigmem()