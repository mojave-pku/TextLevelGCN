import dgl
import torch
import torch.nn.functional as F
import numpy as np
import word2vec


def gcn_msg(edge):
    return {'m': edge.src['h'], 'w': edge.data['w']}


def gcn_reduce(node):
    w = node.mailbox['w']

    new_hidden = torch.mul(w, node.mailbox['m'])

    new_hidden,_ = torch.max(new_hidden, 1)

    node_eta = torch.sigmoid(node.data['eta'])
    # node_eta = F.leaky_relu(node.data['eta'])

    # new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
    # print(new_hidden.shape)

    return {'h': new_hidden}


class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 vocab,
                 n_gram,
                 drop_out,
                 edges_num,
                 edges_matrix,
                 max_length=350,
                 trainable_edges=True,
                 pmi=None,
                 cuda=True
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.vocab = vocab
        # print(len(vocab))
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        print(edges_num)
        print(pmi.shape)

        self.node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)
        
        self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)
            
        self.edges_num = edges_num
        if trainable_edges:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(edges_num, 1), freeze=False)
        else:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)

        self.hidden_size_node = hidden_size_node

        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec('glove.6B.200d.vec.txt')))
        self.node_hidden.weight.requires_grad = True

        self.len_vocab = len(vocab)

        self.ngram = n_gram

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.max_length = max_length

        self.edges_matrix = edges_matrix

        self.dropout = torch.nn.Dropout(p=drop_out)

        self.activation = torch.nn.ReLU()

        self.Linear = torch.nn.Linear(hidden_size_node, class_num, bias=True)

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def load_word2vec(self, word2vec_file):
        model = word2vec.load(word2vec_file)

        embedding_matrix = []

        for word in self.vocab:
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                # print(word)
                embedding_matrix.append(model['the'])

        embedding_matrix = np.array(embedding_matrix)

        return embedding_matrix

    def add_all_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []

        local_vocab = list(set(doc_ids))

        for i, src_word_old in enumerate(local_vocab):
            src = old_to_new[src_word_old]
            for dst_word_old in local_vocab[i:]:
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            # self circle
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                # - first connect the new sub_graph
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            # self circle
            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        local_vocab = set(doc_ids)

        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph()

        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)

        sub_graph.ndata['h'] = local_node_hidden

        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new)

        edges, old_edge_id = [], []
        # edges = []

        edges.extend(seq_edges)

        old_edge_id.extend(seq_old_edges_id)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)

        srcs, dsts = zip(*edges)
        sub_graph.add_edges(srcs, dsts)
        try:
            seq_edges_w = self.seq_edge_w(old_edge_id)
        except RuntimeError:
            print(old_edge_id)
        sub_graph.edata['w'] = seq_edges_w

        return sub_graph

    def forward(self, doc_ids, is_20ng=None):
        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]

        batch_graph = dgl.batch(sub_graphs)

        batch_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            reduce_func=dgl.function.max('weighted_message', 'h')
        )

        h1 = dgl.sum_nodes(batch_graph, feat='h')

        drop1 = self.dropout(h1)
        act1 = self.activation(drop1)

        l = self.Linear(act1)

        return l
