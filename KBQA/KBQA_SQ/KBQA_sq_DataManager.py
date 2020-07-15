# -*- coding: utf-8 -*-

from collections import defaultdict
import random

from tqdm import tqdm
import numpy as np

import sys
import logging
import json
import os
import math
import time
import re
import string
from collections import defaultdict



dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

sys.path.append('../..')
import KBQA_dat.KBQA_file_paths as dt_p
from KBQA_dat import word_vocab_KBQA
from KBQA_dat.data_preprocess.train_data_generate import delte_punctuation_kbqa
from KNNdata.KBQA_sq_KNNData import KBQA_SQ_KNNDataManager

class KBQA_SQ_DataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.alias_q_max_len = self.opt.alias_q_max_len

        self.vocab = word_vocab_KBQA.WordVocab(dt_p.KBQA_SQ_vocab, emb_dim=300)#pad oov start end + simpleQA_vocab embed

        self.kn = KBQA_SQ_KNNDataManager(k = self.opt.knn_nums)
        self.kn.read_knn_list(k = self.opt.knn_nums)
        self.k_value = self.kn.k


        q_fs = [u'{}sq_{}'.format(dt_p.KBQA_SQ_dp, i) for i in 'valid.replace_ne train.replace_ne test'.split(" ")]
        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.KBQA_SQ_re_list).readlines()]
        self.rels2id = self.get_rels2id()
        self.rel_questions = self.get_query_of_rel()
       
        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {}  # 上一个epoch的结果
        self.train_pair_like_i = {}
        self.parsed_train_dat_cache = self.parse_f(q_fs[1])

        self.valid_smp_pairs, self.cdt_num_of_valid_dat = self.a_f2x_valid_and_test(q_fs[0])
        self.test_smp_pairs, self.cdt_num_of_test_dat = self.a_f2x_valid_and_test(q_fs[2])
        self.question_text, self.relation_text, self.relation_num_dat, self.relation_id_dat = self.a_f2x_test_text(q_fs[2])
        self.get_train = True
        self.train_smp_pairs = self.a_f2x(self.q_fs[1])
        self.get_train = False

        ##KBQA test process
        self.test_questions_subject_mids_relations, self.test_qs, self.test_gold_rels = self.read_test_question(q_fs[2])
        self.candidates_mid_relation_midscore_objects, self.test_cdt_rels = self.read_test_candidate(dt_p.KBQA_SQ_entity_link_tuples)

    def get_rels2id(self):
        rels2id={}
        for id in range(len(self.rels)):
            rels2id[self.rels[id]] = id
        return rels2id

    def read_entity_name_list(self):
        mid2_entity = {}
        entity2mid = {}
        train_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_train_entity_name, encoding="utf-8").readlines() for
                            ln_pair
                            in ln.strip('\n').split('\t')]
        valid_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_valid_entity_name, encoding="utf-8").readlines() for
                            ln_pair
                            in ln.strip('\n').split('\t')]
        test_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_test_entity_name, encoding="utf-8").readlines() for ln_pair
                           in ln.strip('\n').split('\t')]

        for i in range(0, len(test_mid_entity), 2):
            mid2_entity[test_mid_entity[i]] = test_mid_entity[i + 1]
            entity2mid[test_mid_entity[i + 1]] = test_mid_entity[i]
        return mid2_entity, entity2mid


    def read_test_question(self, f):
        test_questions_subject_mids_relations = []
        gold_rels = []
        qs = []
        lns = [ln.strip('\n') for ln in open(f, encoding='utf-8').readlines()]

        for ln in lns:
            line_items = ln.split('\t')
            gr = line_items[-3]
            gold_rel = gr
            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r))
                gold_rel = multi_gold_r
            gold_rels.append(gold_rel)
            qs.append(line_items[3].strip())
            test_questions_subject_mids_relations.append([line_items[3].strip(), line_items[0].strip(),
                                                              gold_rel])

        print("len test_questions_subject_mids_relations:", len(test_questions_subject_mids_relations))
        return test_questions_subject_mids_relations, qs, gold_rels

    def read_test_candidate(self, f):

        with open(f, "r", encoding="utf-8") as f:
            candidates_str = json.load(f)
        print("len candidates:", len(candidates_str))

        # mid_subject==relation==object==mid_subject--object.name/alias.name--subject_str==0
        #chen added: relation_id using  self.rels2id
        candidates_mid_relation_midscore_objects = []
        cdt_rels = []
        for candidate in candidates_str:
            mid_relation_midscore_object = []
            cdt_rel = []
            for can in candidate:
                can_items = can.strip().split("==")
                assert len(can_items) > 4
                mid_link = can_items[3]
                mid_link = mid_link.strip().split("--")
                assert len(mid_link) == 3
                mid_relation_midscore_object.append([can_items[0].strip(), can_items[1].strip(), can_items[2].strip(),
                                                     mid_link[0].strip(), mid_link[1].strip(), mid_link[2].strip(),
                                                     float(can_items[-1]),self.rels2id.get(can_items[1].strip())])
                cdt_rel.append(self.rels2id.get(can_items[1].strip()))
            cdt_rels.append(cdt_rel)
            candidates_mid_relation_midscore_objects.append(mid_relation_midscore_object)

        print(len(candidates_mid_relation_midscore_objects))
        print(candidates_mid_relation_midscore_objects[0])
        return candidates_mid_relation_midscore_objects, cdt_rels

    def a_f2x_test_text(self,f):
        gold_rel, cdt_rels, qs = self.parse_f(f)
        cdt_num_of_valid_dat = []
        cdt_name_of_valid_dat = []
        cdt_id_of_valid_dat = []
        question = []
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            q_cdt_rels = [gr] + [r for r in set(cdt_rs) if r != gr]
            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            cdt_id_of_valid_dat.append(q_cdt_rels)
            q_cdt_relname = [self.rels[r] for r in q_cdt_rels]
            question.append(q)
            cdt_name_of_valid_dat.append(q_cdt_relname)
        return question, cdt_name_of_valid_dat,cdt_num_of_valid_dat,cdt_id_of_valid_dat

    @staticmethod
    def words_of_rel(rel):
        """
        """
        #return rel.split('/')[-1].split('_')
        return u'_'.join([rel.split('/')[-2]]).split('_') + rel.split('/')[-1].split('_')
      

    def idxs_of_rel(self, rel_idx):
        """

        :param rel_idx: 
        :return:
        """
        r_ws = self.words_of_rel(self.rels[rel_idx])
        rel_x = self.vocab.seqword2id(r_ws) + [self.vocab.word2id(self.rels[rel_idx])]
        return rel_x


    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parsed_train_dat_cache
        smps = []
        for gr, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)  #word2id
            q_knn_sts = [self.kn.knns[idx].tolist() for idx in q_x]
            gr_x = self.idxs_of_rel(gr)

            gr_text = self.rels[gr] #得到true predicate text
            grx_knn_sts = [self.kn.knns[idx].tolist() for idx in gr_x]

            for r in cdt_rs:
                if r == gr:
                    continue
                r_x = self.idxs_of_rel(r)
                rx_knn_sts = [self.kn.knns[idx].tolist() for idx in r_x]

                smps.append((q_x, gr_x, r_x,grx_knn_sts, rx_knn_sts,q_knn_sts)) #3-5
        return smps

    def a_f2x_valid_and_test(self, f):
        """
        :param f:
        :return:
        """
        gold_rel, cdt_rels, qs = self.parse_f(f)
        smps = []
        cdt_num_of_valid_dat = []
        # question_num = 0
        # candidate_num = 0
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            q_x = self.rep_of_question(q)
            q_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in q_x]

            q_cdt_rels = [gr] + [r for r in set(cdt_rs) if r != gr]

            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            # question_num += 1
            # candidate_num += len(q_cdt_rels)
            for r_idx in q_cdt_rels:
                rx1 = self.idxs_of_rel(r_idx)
                rx1_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in rx1]

                smps.append((q_x, rx1,rx1_knn_valid_test_sts,q_knn_valid_test_sts))
            #     r_alias_q_x = self.rep_of_alias_questions(r_idx, q, for_valid=True)
            #     smps.append((q_x, rx1, r_alias_q_x))
        return smps, cdt_num_of_valid_dat

    def get_query_of_rel(self):
        rel_questions = defaultdict(list)
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for gr, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            rel_questions[gr].append(q)

        rel_q_cnt = {rel: len(qs) for rel, qs in rel_questions.items()}
        print(sum([cnt for rel, cnt in rel_q_cnt.items()]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt < 10]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt < 5]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt >= 10]))
        return rel_questions

    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())


    # ############################################ padding ############################
    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
            if max_len==1:
                max_len = max_len + 1
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [0] * (max_len - len(idxs))
            idxs_batch_padded.append(idxs + to_add)
        return idxs_batch_padded

    def dynamic_padding_for_knn_idxs(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
            if max_len==1:
                max_len = max_len + 1
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [[0]* (self.k_value-1)] * (max_len - len(idxs))
            idxs_batch_padded.append(idxs + to_add)
        return idxs_batch_padded

    def dynamic_padding_train_batch(self, batch):
        """
        :param batch:
        :return:
        """
        # q_idxs, g_rel_idxs, g_rel_alias_q_idxs, cdt_rel_idxs, cdt_rel_alias_q_idxs = batch
        q_idxs, g_rel_idxs, cdt_rel_idxs, g_knn_sts,negknn_sts, qknn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_rel_idxs = self.dynamic_padding(g_rel_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        g_knn_sts = self.dynamic_padding_for_knn_idxs(g_knn_sts)
        negknn_sts = self.dynamic_padding_for_knn_idxs(negknn_sts)
        qknn_sts = self.dynamic_padding_for_knn_idxs(qknn_sts)

        batch = q_idxs, g_rel_idxs, cdt_rel_idxs,g_knn_sts, negknn_sts, qknn_sts # 这个alias 可是不得了。不能用的。

        return list(np.array(i) for i in batch)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, r_knn_sts, q_knn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        r_knn_sts = self.dynamic_padding_for_knn_idxs(r_knn_sts)
        q_knn_sts = self.dynamic_padding_for_knn_idxs(q_knn_sts)

        batch = q_idxs, cdt_rel_idxs,r_knn_sts, q_knn_sts   # , cdt_rel_alias_q_idxs
        return list(np.array(i) for i in batch)

    # ############################################ batches ############################
    def get_train_batchs(self, epoch, for_keras=False):
        self.train_smp_pair_num = len(self.train_smp_pairs)
        print(self.train_smp_pair_num, 'training sample num')
        # Q1, true_p1, neg_p1
        # Q1, true_p1, neg_p2
        # Q2, true_p1, neg_p1...

        indices = list(range(self.train_smp_pair_num))
        # if epoch < 1:
        random.shuffle(indices) #shuffle
        batch = [[] for i in range(6)]
        idx = 0
        batch_indices = []

        if epoch > 1 and self.opt.with_selection:
            #SimpleQuestion selected
            indices = self.select_smps(epoch)
        # random.shuffle(indices)

        self.train_pair_like_i_1 = {k: v for k, v in self.train_pair_like_i.items()}

        while idx < len(indices):
            items = self.train_smp_pairs[indices[idx]]
            batch_indices.append(idx)
            for i, item in enumerate(items):
                batch[i].append(item)
            if len(batch[0]) == self.opt.train_batchsize or idx + 1 == self.train_smp_pair_num:
                batch = self.dynamic_padding_train_batch(batch)
                self.train_batch_indices_cache = batch_indices
                if for_keras:
                    batch[2] = batch[2].reshape(batch[2].shape[0], batch[2].shape[1] * batch[2].shape[2])
                    batch[4] = batch[4].reshape(batch[4].shape[0], batch[4].shape[1] * batch[4].shape[2])
                yield batch
                batch = [[] for i in range(6)]
                batch_indices = []
            idx += 1

    def select_smps(self, epoch):
        indices = list(range(len(self.train_smp_pairs)))
        diffs = []  
        last_indices = []
        need_review_indices = []
        for idx in indices:
            if idx not in self.train_pair_like_i:
                self.train_pair_like_i[idx] = self.train_pair_like_i_1[idx]
            if self.train_pair_like_i[idx] > 0:
                last_indices.append(idx)
                cost_i = self.train_pair_like_i[idx]
                diffs.append(cost_i)
            else:
                need_review_indices.append(idx)

        min_dif = min(diffs)
        max_dif = max(diffs)
        diffs = (np.array(diffs) - min_dif) / (max_dif - min_dif)

        # indices_selected = pd.Series(last_indices).sample(frac=0.8, weights=diffs, replace=False)
        indices_selected = last_indices
        indices_selected = list(indices_selected)

        review_indices = random.sample(need_review_indices, k=int(self.opt.review_rate * len(need_review_indices)))
        # review_indices = random.sample(need_review_indices, k=int(0.3 * len(need_review_indices)))

        assert len(indices_selected) == len(set(indices_selected))

        if epoch > 10:
            # random.shuffle(indices_selected)
            random.shuffle(review_indices)
            final_indices = indices_selected + review_indices
        else:
            final_indices = indices
            random.shuffle(final_indices)
        # remove_dup_idxs

        # return sorted(indices, key=lambda idx: self.train_pair_like_i[idx], reverse=True)
        print(f'selected indices={len(indices_selected)}, review indices={len(review_indices)}')

        return final_indices

    def valid_or_test_batches(self, valid_or_test='valid', for_keras=False):
        the_smps = self.valid_smp_pairs if valid_or_test == 'valid' else self.test_smp_pairs
        smp_pair_num = len(the_smps)
        #batch = [[] for i in range(4)]
        batch = [[] for i in range(4)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                if for_keras:
                    batch[2] = batch[2].reshape(batch[2].shape[0], batch[2].shape[1] * batch[2].shape[2])
                yield batch
                #batch = [[] for i in range(4)]
                batch = [[] for i in range(4)]

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

    @staticmethod
    def parse_f(fnm):
        """
        :param fnm:
        :return:
        """
        lns = [ln.strip('\n') for ln in open(fnm, encoding='utf-8').readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []
        print(len(lns), fnm)
        for ln in lns:
            #gr, cdt_rs, q = ln.split('\t')
            gr, cdt_rs = ln.split('\t')[-3:-1]
            q  = ln.split('\t')[3]
            qs.append(q)
            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r))  
                gold_rel.append(multi_gold_r)
            else:
                gold_rel.append(int(gr))

            if cdt_rs == 'noNegativeAnswer':
                cdt_rels.append([])
            else:
                cdt_rels.append([int(i) for i in cdt_rs.split()])
        return gold_rel, cdt_rels, qs

'''

    def kbqa_test_batches(self):
        the_smps = self.test_q_rel_pair
        batch = [[] for i in range(2)]
        for idx0, smp_pairs in enumerate(the_smps):
            for idx, smp in enumerate(smp_pairs):
                for i, item in enumerate(smp):
                    batch[i].append(item)
            if idx0 < 5:
                print(batch)

            batch = self.dynamic_padding_valid_batch(batch)
            yield batch
            batch = [[] for i in range(2)]
            
    def test_for_kbqa(self):
        test_questions_subject_mids_relations = self.test_questions_subject_mids_relations
        candidates_mid_relation_midscore_objects = self.candidates_mid_relation_midscore_objects

        assert len(test_questions_subject_mids_relations) == len(candidates_mid_relation_midscore_objects)

        smps = []
        for q_idx, (gold_relations, candidate_mid_relation, q, gr, cdt_rs) in enumerate(zip(test_questions_subject_mids_relations,
                                                         candidates_mid_relation_midscore_objects,self.test_qs,self.test_gold_rels,
                                                         self.test_cdt_rels)):
            if len(candidate_mid_relation) == 0:
                continue
            smp = []
            assert len(candidate_mid_relation) == len(cdt_rs)

            for candi_relation, r_idx in zip(candidate_mid_relation, cdt_rs):
                q_withen = delte_punctuation_kbqa(q).replace(candi_relation[5], '#head_entity#')
                q_x = self.rep_of_question(q_withen)
                rx1 = self.idxs_of_rel(r_idx)
                smp.append((q_x, rx1))
            smps.append(smp)

        return smps
'''
