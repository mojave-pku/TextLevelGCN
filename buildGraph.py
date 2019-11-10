import dgl
import torch

class GraphBuilder(object):
    def __init__(self, words, hiddenSizeNode):
        self.graph = dgl.DGLGraph()
        self.word2id = dict(zip(words, range(len(words))))
        self.graph.add_nodes(len(words))

        # add hidden para for nodes.
        self.graph.ndata['h'] = torch.nn.Parameter(
            torch.Tensor(len(words), hiddenSizeNode)
        )

        # all node are supposed to connected.
        # warning: self-connected enabled.
        for i in range(len(words)):
            self.graph.add_edges(i, range(0, len(words)))

        # add hidden para for edges. Only edge weight (size = 1 )
        self.graph.edata['h'] = torch.nn.Parameter(
            torch.Tensor(self.graph.number_of_edges(), 1)

        )


if __name__ == '__main__':
    pass
