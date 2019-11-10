from nltk.tokenize import word_tokenize
import random,re, os, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"[0-9]", " ", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def remove_short(string):
    results = []
    for word in string.split(' '):
        if len(word) < 3:
            continue
        else:
            results.append(word)

    return ' '.join(results)


def stem_corpus():
    stemmer = WordNetLemmatizer()

    with open('data/mr/text_train.txt') as f:
        raw_text = f.read()

    with open('data/mr/label_train.txt') as f:
        raw_labels = f.read()

    labels = []
    for raw_label in raw_labels.split('\n'):
        if raw_label == '1':
            labels.append('pos')
        elif raw_label == '0':
            labels.append('neg')
        else:
            if len(raw_label) == 0:
                continue
            raise ValueError(raw_label)

    corpus = raw_text.split('\n')
    corpus = [clean_str(doc) for doc in corpus]
    corpus = [remove_short(doc) for doc in corpus]

    tokenized_corpus = [word_tokenize(doc) for doc in corpus]

    results = []

    for line in tokenized_corpus:
        results.append(' '.join([stemmer.lemmatize(word) for word in line]))

    results = list(zip(labels, results))
    results = ['\t'.join(line) for line in results]
    random.shuffle(results)

    with open('data/mr/mr-train-stemmed.txt', 'w') as f:
        f.write('\n'.join(results))


def cut_datasets():
    for dataset in ['r8', 'r52', 'oh']:
        with open(os.path.join('.', 'data', dataset, dataset+'-stemmed.txt')) as f:
            all_cases = f.read().split('\n')
            print('datasets: ', dataset, ', total length:', len(all_cases))
            cut_index = int(len(all_cases) * 0.9)
            train_cases = all_cases[:cut_index]
            dev_cases = all_cases[cut_index+1:]

        with open(os.path.join('.', 'data', dataset, dataset+'-train-stemmed.txt'), 'w') as f:
            f.write('\n'.join(train_cases))
        with open(os.path.join('.', 'data', dataset, dataset+'-dev-stemmed.txt'), 'w') as f:
            f.write('\n'.join(dev_cases))


class Ohsumed(object):
    def __init__(self):
        self.base = './ohsumed-first-20000-docs'

    def cal_freq(self):
        results = {}
        for _, _, file_names in os.walk(self.base):
            for file in file_names:
                if file not in results.keys():
                    results[file] = 1
                else:
                    results[file] += 1

        single_target = []
        for file in results.keys():
            if results[file] == 1:
                single_target.append(file)

        with open('ohsumed-single-index.txt', 'w') as f:
            f.write(','.join(single_target))

        print(len(single_target))

    def make_set(self):
        set = 'train'
        current = os.path.join(self.base, set)
        result = []

        with open('ohsumed-single-index.txt') as f:
            raw = f.read()
            indexs = raw.split(',')

        for dir, _, file_names in os.walk(current):
            if len(file_names) == 0:
                continue
            type = dir[-3:]
            for file in file_names:
                if file in indexs:
                    with open(os.path.join(dir, file)) as f:
                        text = f.read()
                        text = text.replace('\n', ' ')
                        text = self.clean_text(text)
                        result.append('\t'.join([type, text]))

        random.shuffle(result)

        with open('train-shuffled.txt','w') as f:
            f.write('\n'.join(result))

    @staticmethod
    def clean_text(text):
        # stop_words = stopwords.words('english')
        stop_words = []
        stop_words.extend(['!', ',' ,'.' ,'?' ,'-s' ,'-ly' ,'</s> ', 's'])
        stemmer = WordNetLemmatizer()

        text = remove_short(text)
        text = clean_str(text)

        text = word_tokenize(text)

        text = [word for word in text if word not in stop_words]

        text = [stemmer.lemmatize(word) for word in text]

        return ' '.join(text)



if __name__ == '__main__':
    # stem_corpus()
    # shuffle_20ng()
    # o = Ohsumed()
    # o.make_set()
    # nltk.download('wordnet')

    # clean_mr()
    cut_datasets()