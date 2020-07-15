# -*- coding: utf-8 -*-

import torch.nn as nn
import sys
import os

dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

dev = 1

import numpy as np
import torch

import sys
import os


dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

#sys.path.append('../..')
import KBQA_dat.word_vocab_KBQA as word_vocab
import KBQA_dat.KBQA_file_paths as dt_p

class KBQA_SQ_KNNDataManager:
    def __init__(self, K = 8, k=8):
        self.vocab = word_vocab.WordVocab(dt_p.KBQA_SQ_vocab, emb_dim=300)
        self.K = K
        self.k = k

    def get_knn_list(self):

        cos = nn.CosineSimilarity(dim=1)
        val2_new = np.reshape(self.vocab.vs[4:], (-1, 300))
        length = len(self.vocab.vs[4:])
        print("length is ",length)
        print("half_length is ",int(length/2))

        self.knn_list_0 = [cos(torch.tensor(np.reshape(val1[1],(1,300))), torch.tensor(val2_new)).tolist() for val1 in enumerate(self.vocab.vs[4:int(length/2)])]

        idx_0 = np.argsort(self.knn_list_0,axis=1) #shape array (2,2)
        idx_0 = idx_0 + 4
        idx_new_0 = idx_0[:,-50:]  #we need to read the file and append 4 0-vectors in first 4 lines

        np.savetxt(dt_p.k50_kbqa_sq_f_0, idx_new_0, fmt='%d')
        #np.savetxt(dt_p.k50_kbqa_sq_f_1, idx_new_1, fmt='%d')

    def read_knn_list(self, k = 8):
        knn_file = dt_p.k8_kbqa_sq_f
        self.knns = []

        with open(knn_file, 'r', encoding='utf-8') as knn_f:
            lns = [ln.strip('\n') for ln in knn_f.readlines()]
        for ln in lns:

            vs = [int(v) for v in ln.split()]
            self.knns.append(vs[-k:-1])
            #self.knns.append(vs[1:k+1])
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

    KNN = KBQA_SQ_KNNDataManager(K=8)
    KNN.get_knn_list()
    #KNN.read_knn_list(k=8)
    #KNN.test_vocab()

