# -*- coding: utf-8 -*-

import word_vocab
import dat.file_pathes as dt_p
import torch.nn as nn
import torch
import numpy as np

class WebQAKNNDataManager:
    def __init__(self, K = 8, k=8):
        self.vocab = word_vocab.WordVocab(dt_p.webqa_vocab_f, emb_dim=300)
        self.K = K
        self.k = k

    def get_knn_list(self):

        cos = nn.CosineSimilarity(dim=1)
        val2_new = np.reshape(self.vocab.vs[4:], (-1, 300))
        self.knn_list = [cos(torch.tensor(np.reshape(val1[1],(1,300))), torch.tensor(val2_new)).tolist() for val1 in enumerate(self.vocab.vs[4:])]

        idx = np.argsort(self.knn_list,axis=1) #shape array (2,2)
        idx = idx + 4
        idx_new = idx[:,-self.K:]  #we need to read the file and append 4 0-vectors in first 4 lines

        np.savetxt(dt_p.k8_webqa_f, idx_new, fmt='%d')

    def read_knn_list(self, k = 8):
        knn_file = dt_p.k8_webqa_f
        self.knns = []

        with open(knn_file, 'r', encoding='utf-8') as knn_f:
            lns = [ln.strip('\n') for ln in knn_f.readlines()]
        for ln in lns:
            vs = [int(v) for v in ln.split()]
            self.knns.append(vs[-k:-1])

        b = np.array([[0]*(k-1),[1]*(k-1),[2]*(k-1),[3]*(k-1)])
        self.knns = np.insert(self.knns, 0, values=b, axis=0)
        print(self.knns)
        self.knns = np.asarray(self.knns)

    def test_vocab(self):

        knn_ids = self.knns[100:180]
        all_words = []

        for ids in knn_ids:
            words = self.vocab.id2seqword(ids)
            print(words)
            all_words.append(words)

if __name__ == '__main__':
    KNN = WebQAKNNDataManager(K=8)
    #KNN.get_knn_list()
    KNN.read_knn_list(k=8)
    KNN.test_vocab()
