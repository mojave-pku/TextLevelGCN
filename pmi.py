from data_helper import DataHelper
import numpy as np
import torch

def cal_PMI(dataset: str, window_size=20):
    helper = DataHelper(dataset=dataset, mode="train")
    content, _ = helper.get_content()
    pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    word_count =np.zeros(len(helper.vocab), dtype=int)
    
    for sentence in content:
        sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            try:
                word_count[helper.d[word]] += 1
            except KeyError:
                continue
            start_index = max(0, i - window_size)
            end_index = min(len(sentence), i + window_size)
            for j in range(start_index, end_index):
                if i == j:
                    continue
                else:
                    target_word = sentence[j]
                    try:
                        pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                    except KeyError:
                        continue
        
    total_count = np.sum(word_count)
    word_count = word_count / total_count
    pair_count_matrix = pair_count_matrix / total_count
    
    pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            pmi_matrix[i, j] = np.log(
                pair_count_matrix[i, j] / (word_count[i] * word_count[j]) 
            )
    
    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            if pmi_matrix[i, j] != 0:
                edges_weights.append(pmi_matrix[i, j])
                edges_mappings[i, j] = count
                count += 1

    edges_weights = np.array(edges_weights)

    edges_weights = edges_weights.reshape(-1, 1)
    # print(edges_weights.shape)
    edges_weights = torch.Tensor(edges_weights)
    
    return edges_weights, edges_mappings, count


if __name__ == "__main__":
    cal_PMI('r8')