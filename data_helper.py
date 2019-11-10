import os
import torch
import csv


class DataHelper(object):
    def __init__(self, dataset, mode='train', vocab=None):
        allowed_data = ['r8', '20ng', 'r52', 'mr', 'oh']

        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset

        self.mode = mode

        self.base = os.path.join('data', self.dataset)

        self.current_set = os.path.join(self.base, '%s-%s-stemmed.txt' % (self.dataset, self.mode))

        with open(os.path.join(self.base, 'label.txt')) as f:
            labels = f.read()
        self.labels_str = labels.split('\n')

        content, label = self.get_content()

        self.label = self.label_to_onehot(label)
        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

    def label_to_onehot(self, label_str):

        return [self.labels_str.index(l) for l in label_str]

    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]
        if self.dataset == '20ng' or 'r52':
            cleaned = []
            for i, pair in enumerate(content):
                if len(pair) < 2:
                    # print(i, pair)
                    pass
                else:
                    cleaned.append(pair)
        else:
            cleaned = content

        label, content = zip(*cleaned)

        return content, label

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab-5.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def build_vocab(self, content, min_count=10):
        vocab = []

        for c in content:
            words = c.split(' ')
            for word in words:
                if word not in vocab:
                    vocab.append(word)

        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        results = []
        for word in freq.keys():
            if freq[word] < min_count:
                continue
            else:
                results.append(word)

        results.insert(0, 'UNK')
        with open(os.path.join(self.base, 'vocab-5.txt'), 'w') as f:
            f.write('\n'.join(results))

        self.vocab = results

    def count_word_freq(self, content):
        freq = dict(zip(self.vocab, [0 for i in range(len(self.vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        with open(os.path.join(self.base, 'freq.csv'), 'w') as f:
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                yield content, torch.tensor(label).cuda(), i


if __name__ == '__main__':
    data_helper = DataHelper(dataset='r8')
    content, label = data_helper.get_content()
    data_helper.build_vocab(content)
