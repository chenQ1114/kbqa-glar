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

sys.path.append('../..')
import KBQA_dat.word_vocab_KBQA as word_vocab
import KBQA_dat.KBQA_file_paths as dt_p

class KBQA_WEBQA_KNNDataManager:
    def __init__(self, vocab_path, train_save_path, K = 8, k=8):
        self.vocab = word_vocab.WordVocab(vocab_path, emb_dim=300)
        self.train_save_path = train_save_path
        self.K = K
        self.k = k


    def get_knn_list(self):
        cos = nn.CosineSimilarity(dim=1)
        val2_new = np.reshape(self.vocab.vs[4:], (-1, 300))

        knn_list =[]
        with open(self.train_save_path, "a+", encoding="utf-8") as f:
            for index,value in enumerate(self.vocab.vs[16415:]):
                knn_list.append(cos(torch.tensor(np.reshape(value, (1, 300))), torch.tensor(val2_new)).tolist())
                id_train = np.argsort(knn_list, axis=1)  # shape array (2,2)
                id_train = id_train + 4
                idx_train_new = id_train[:, -20:]

                if index%10==0 and index!=0:
                    for line in idx_train_new.tolist():
                        f.write(' '.join([str(m) for m in line])+"\n")
                    knn_list =[]

            id_train = np.argsort(knn_list, axis=1)  # shape array (2,2)
            id_train = id_train + 4
            idx_train_new = id_train[:, -20:]
            for line in idx_train_new.tolist():
                f.write(' '.join([str(m) for m in line])+"\n")

        print("The KNN calculation is done.")

    def read_knn_list(self, k = 8):
        knn_file = self.train_save_path
        train_knns = []
        with open(knn_file, 'r', encoding='utf-8') as knn_f:
            lns = [ln.strip('\n') for ln in knn_f.readlines()]
        for ln in lns:
            vs = [int(v) for v in ln.split()]
            train_knns.append(vs[-k:-1])
            # self.knns.append(vs[1:k+1])
        b = np.array([[0] * (k - 1), [1] * (k - 1), [2] * (k - 1), [3] * (k - 1)])
        train_knns = np.insert(train_knns, 0, values=b, axis=0)
        train_knns = np.asarray(train_knns)
        return train_knns

    def test_vocab(self):

        knn_ids = self.knns[100:180]
        all_words = []

        for ids in knn_ids:
            words = self.vocab.id2seqword(ids)
            print(words)
            all_words.append(words)
    def test(self):
        text = 'operation' #10347
        id = self.vocab._word_to_id.get(text)
        print(id)
        knn_list = self.read_knn_list()
        print(knn_list[id])
        for m in knn_list[id]:
            print(self.vocab._id_to_word[m])


if __name__ == '__main__':
    
    KNN = KBQA_WEBQA_KNNDataManager(vocab_path = dt_p.KBQA_Web_mix_vocab,train_save_path= dt_p.kbqa_web_mix_knn_f, K=8)
    KNN.get_knn_list()
    # test_knn_list = KNN.read_knn_list(train_or_test='train',k=8)
