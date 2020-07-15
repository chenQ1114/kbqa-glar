import random

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

import word_vocab
import KBQA_dat.KBQA_file_paths as dt_p
from KBQA_dat import word_vocab_KBQA
from KNNdata.KBQA_webqa_KNNData import KBQA_WEBQA_KNNDataManager

class KBQA_Webqa_DataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.vocab = word_vocab_KBQA.WordVocab(dt_p.KBQA_Web_coling_ture_whole_vocab,
                                               emb_dim=300)  
        self.kn = KBQA_WEBQA_KNNDataManager(vocab_path=dt_p.KBQA_Web_coling_ture_whole_vocab, train_save_path= dt_p.kbqa_web_coling_true_whole_knn_f,k = self.opt.knn_nums)
        self.train_knn_list = self.kn.read_knn_list(k = self.opt.knn_nums)
        self.k_value = self.kn.k

        q_fs = [u'{}GnnQA.RE.true_whole.{}.txt'.format(dt_p.KBQA_Web_coling_true_whole_dp, i) for i in 'train test'.split()]
        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.KBQA_Web_coling_true_whole_re_list).readlines()]


        self.rels2id = self.get_rels2id()

        self.rel_questions = self.get_query_of_rel()

        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {}  
        self.train_pair_like_i = {}


        self.test_smp_pairs, self.cdt_num_of_test_dat = self.a_f2x_valid_and_test(q_fs[1])

        self.question_text, self.relation_text, self.relation_num_dat, self.relation_id_dat = self.a_f2x_test_text(
            q_fs[1])
        self.gold_relation_text = self.a_f2x_text_gold_rels(q_fs[1])
        self.smps_rr = self.a_f2x(self.q_fs[0])


        self.test_questions_subject_mids_relations, self.test_qs, self.test_gold_subjects, self.test_gold_rels, self.test_objects, self.test_gold_rels_mentions, self.test_q_indexs = self.read_test_question(q_fs[-1])
        self.candidates_mid_relation_midscore_objects, self.candidate_cdt_rels = self.read_test_candidate(dt_p.KBQA_Web_coling_test_entity_linking_true_whole_update)

        self.ever_contribute = set()

    def get_rels2id(self):
        rels2id={}
        for id in range(len(self.rels)):
            rels2id[self.rels[id]] = id
        return rels2id

    def get_query_of_rel(self):
        rel_questions = defaultdict(list)
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        for grs, cdt_rs, q in zip(gold_rel, cdt_rels, qs):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        for r, qs in rel_questions.items():
            rel_questions[r] = list(set(qs))
        return rel_questions

    @staticmethod
    def parse_f(fnm, silent=False):
        """
        simpleQuestion
        :param fnm:
        :return:
        """
        lns = [ln.strip('\n') for ln in open(fnm, encoding='utf-8').readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []

        if not silent:
            print(len(lns), fnm)
        for ln in lns:
            q, gr, cdt_rs = ln.split('\t')[2:5]

            qs.append(q.strip())

            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r))   
                gold_rel.append(multi_gold_r)
            else:
                gold_rel.append(int(gr))

            if cdt_rs == '':
                # print("the negative relation is null in question: "+q + '\t'+ fnm)
                cdt_rels.append([])
            else:
                cdt_rels.append([int(i) for i in cdt_rs.split()])

        return gold_rel, cdt_rels, qs


    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())


    def words_of_a_rel(self, rel):
        """
        """
        return rel.split('/')[-1].split('_')

        # return u'_'.join([rel.split('/')[-2]]).split('_') + rel.split('/')[-1].split('_')

    def words_of_web_rel(self, rel_idx):

        rel = self.rels[rel_idx]
        ws = []
        rels = []
        for sub_rel in rel.split('..'):
            tmp_sub_rel = '/' + sub_rel.replace('.', '/')
            rels.append(tmp_sub_rel)
            ws += self.words_of_a_rel(tmp_sub_rel)

        # if the gold relation is the format of a..b, adding the rels directly. If the len of gold relation is 1, setting the rels as: RELPAD..b
        while len(rels) < 2:
             rels = [self.vocab.RELPAD] + rels
        if len(rels) > 2:
            rels = rels[:2]
        if not self.opt.use_wh_hot:
            return ws + rels
        if self.get_train and random.randint(1, 100) > 10:
            return ['m'] + ws + rels
        return ws + rels

    def idxs_of_rel(self, rel_idx):
        """

        :param rel_idx: id in self.rels
        :return:
        """
        return self.vocab.seqword2id(self.words_of_web_rel(rel_idx))

    def a_f2x_valid_and_test(self, f):
        """
        :param f:
        :return:
        """
        # return gold_rel  [100,400]
        #    cdt_rels   [[11 442],[1079 11 254 101]]
        #      qs       [what series does the episode #head_entity# belong to, what book did #head_entity# publish]
        gold_rel, cdt_rels, qs = self.parse_f(f)

        print('test questions', len(qs))
        smps = []
        cdt_num_of_valid_dat = []
        test_gold_rels = []
        cdt_rels_of_q = []
        # question_num = 0
        # candidate_num = 0
        for q_idx, (gr, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)
            q_knn_valid_test_sts = [self.train_knn_list[idx].tolist() for idx in q_x]

            grs = (gr if isinstance(gr, list) else [gr])
            test_gold_rels.append(grs)
            neg_cdts = [r for r in set(cdt_rs) if r not in grs]
            neg_cdts = sorted(neg_cdts)
            q_cdt_rels = grs + neg_cdts

            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            tmp_cdt_rels = []
            # question_num += 1
            # candidate_num += len(q_cdt_rels)
            for r_idx in q_cdt_rels:
                rx1 = self.idxs_of_rel(r_idx)
                rx1_knn_valid_test_sts = [self.train_knn_list[idx].tolist() for idx in rx1]
                tmp_cdt_rels.append(r_idx)

                smps.append((q_x, rx1,rx1_knn_valid_test_sts,q_knn_valid_test_sts))
            cdt_rels_of_q.append(tmp_cdt_rels)
            #     r_alias_q_x = self.rep_of_alias_questions(r_idx, q, for_valid=True)
            #     smps.append((q_x, rx1, r_alias_q_x))
        self.test_gold_rels = test_gold_rels
        self.test_gold_num_of_rels = [len(cdts) for cdts in test_gold_rels]
        self.test_cdt_num_of_rels = [len(cdts) for cdts in cdt_rels_of_q]
        return smps, cdt_num_of_valid_dat

    def a_f2x_test_text(self,f):
        gold_rel, cdt_rels, qs = self.parse_f(f)

        cdt_num_of_valid_dat = []
        cdt_name_of_valid_dat = []
        cdt_id_of_valid_dat = []

        test_gold_rels=[]
        question = []
        for q_idx, (gr, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            grs = (gr if isinstance(gr, list) else [gr])
            test_gold_rels.append(grs)
            neg_cdts = [r for r in set(cdt_rs) if r not in grs]
            neg_cdts = sorted(neg_cdts)
            q_cdt_rels = grs + neg_cdts
            flags = ['g'] * len(grs) + ['n'] * len(neg_cdts)

            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            cdt_id_of_valid_dat.append(q_cdt_rels)
            q_cdt_relname = [self.rels[r] for r in q_cdt_rels]

            question.append(q)
            cdt_name_of_valid_dat.append(q_cdt_relname)

        return question, cdt_name_of_valid_dat,cdt_num_of_valid_dat,cdt_id_of_valid_dat

    def a_f2x_text_gold_rels(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f)
        gold_name_of_test = []

        for q_idx, (gr, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            grs = (gr if isinstance(gr, list) else [gr])
            q_cdt_relname = [self.rels[r] for r in grs]
            gold_name_of_test.append(q_cdt_relname)
        return gold_name_of_test

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f, silent=True)
        self.get_train = True
        more_than_one_golds = {}
        smps_rr = []
        gt_1 = 0
        print('training questions',len(qs))

        for q_idx, (grs, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)  #word2id
            q_knn_sts = [self.train_knn_list[idx].tolist() for idx in q_x]
            if not isinstance(grs, list):
                grs = [grs]
            dup_q = defaultdict(list)
            for gr in grs:
                gr_x = self.idxs_of_rel(gr)
                grx_knn_sts = [self.train_knn_list[idx].tolist() for idx in gr_x]
                for r in cdt_rs:
                    if r in grs:
                        continue
                    smps_rr_idx = len(smps_rr)
                    dup_q[gr].append(smps_rr_idx)
                    r_x = self.idxs_of_rel(r)
                    rx_knn_sts = [self.train_knn_list[idx].tolist() for idx in r_x]
                    smps_rr.append((q_x, gr_x, r_x, grx_knn_sts, rx_knn_sts, q_knn_sts))  # 3-5
            if len(grs) > 1:
                gt_1 += 1
            more_than_one_golds[q_idx] = dup_q
        self.get_train = False
        self.more_than_one_golds = more_than_one_golds
        return smps_rr

    def read_test_question(self, f):
        test_questions_subject_mids_relations = []
        lns = [ln.strip('\n') for ln in open(f, encoding='utf-8').readlines()]
        qs = []
        subjects= []
        gold_rels = []
        gold_rels_mentions = []
        q_indexs =[]
        objects = []


        for ln in lns:
            q, gr, cdt_rs = ln.split('\t')[2:5]
            s = ln.split('\t')[0]
            qs.append(q.strip())

            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r))   
                gold_rels.append(multi_gold_r)
                gold_rel_mention = [self.rels[id] for id in multi_gold_r]
                gold_rel = multi_gold_r
            else:
                gold_rels.append(int(gr))
                gold_rel_mention = self.rels[int(gr)]
                gold_rel = int(gr)
            if len(s.split()) >1:
                multi_s_mid = []
                for m in s.split():
                    multi_s_mid.append(m)
                subject = multi_s_mid
                subjects.append(subject)
            else:
                subjects.append(s.strip())
                subject = s

            q_indexs.append(int(ln.split('\t')[-2]))
            object = [[m.strip() for m in ob.split()] for ob in ln.split('\t')[-1].split('---')]
            objects.append(object)

            gold_rels_mentions.append(gold_rel_mention)

            test_questions_subject_mids_relations.append([q.strip(), subject,
                                                          gold_rel, object,gold_rel_mention])

        return test_questions_subject_mids_relations, qs, subjects, gold_rels,objects,gold_rels_mentions, q_indexs

    def read_test_candidate(self, f):
        websq_test_entity_linking = [ln.strip('\n') for ln in open(f, encoding='utf-8').readlines()]

        candidates_mid_relation_midscore_objects = []
        cdt_rels = []
        q_indexs = []

        mid2_entity = {}

        for index, candidate in enumerate(websq_test_entity_linking):
            q_index = int(candidate.strip().split('=\t=')[0].strip().split('==')[-1])
            q_indexs.append(q_index)

            mid_relation_midscore_object = []
            cdt_rel = []
            for can in candidate.strip().split('=\t=')[:-1]:

                can_items = can.strip().split("==")
                assert len(can_items) == 5
                if can_items[0].strip() not in mid2_entity.keys() and len(can_items[3].split('---'))==2:
                    mid2_entity[can_items[0].strip()] = can_items[3].strip().split('---')[1]

                o_mid_list =[object.strip() for object in can_items[2].split('---')]
                candi_example = [can_items[0].strip(), can_items[1].strip(), o_mid_list,can_items[3].strip().split('---')[1]
                                                     , int(can_items[-1]), self.rels2id.get(can_items[1].strip())]
                if candi_example not in mid_relation_midscore_object:
                    mid_relation_midscore_object.append(candi_example)
                    cdt_rel.append(self.rels2id.get(can_items[1].strip()))

            cdt_rels.append(cdt_rel)
            candidates_mid_relation_midscore_objects.append(mid_relation_midscore_object)
        self.mid2_entity_in_test_entity_linking = mid2_entity
       
        #[['m.03_r3', 'location/country/languages_spoken', ['m.01428y', 'm.04ygk0'], 'Jamaica', 0, 62], ['m.03_r3', 'location/country/official_language', ['m.01428y'], 'Jamaica', 0, 49]
        return candidates_mid_relation_midscore_objects, cdt_rels

    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [0] * (max_len - len(idxs))
            idxs_batch_padded.append(idxs + to_add)
        return idxs_batch_padded

    def dynamic_padding_for_knn_idxs(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [[0]* (self.k_value-1)] * (max_len - len(idxs))
            idxs_batch_padded.append(idxs + to_add)
        return idxs_batch_padded

    def dynamic_padding_train_batch(self, batch):
        """
        g_rel_alias_q_idxs dims: batch * question_num * question_len  
        :param batch:
        :return:
        """
        q_idxs, g_rel_idxs, cdt_rel_idxs, g_knn_sts,negknn_sts, qknn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_rel_idxs = self.dynamic_padding(g_rel_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        g_knn_sts = self.dynamic_padding_for_knn_idxs(g_knn_sts)
        negknn_sts = self.dynamic_padding_for_knn_idxs(negknn_sts)
        qknn_sts = self.dynamic_padding_for_knn_idxs(qknn_sts)

        batch = q_idxs, g_rel_idxs, cdt_rel_idxs,g_knn_sts, negknn_sts, qknn_sts

        return list(np.array(i) for i in batch)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, r_knn_sts, q_knn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        r_knn_sts = self.dynamic_padding_for_knn_idxs(r_knn_sts)
        q_knn_sts = self.dynamic_padding_for_knn_idxs(q_knn_sts)

      
        batch = q_idxs, cdt_rel_idxs,r_knn_sts, q_knn_sts   # , cdt_rel_alias_q_idxs
        return list(np.array(i) for i in batch)

    def get_train_batchs(self, epoch):
        self.smps_rr = self.a_f2x(self.q_fs[0])
        train_smp_pairs = self.smps_rr
        train_smp_pair_num = len(train_smp_pairs)
        indices = list(range(train_smp_pair_num))

        random.shuffle(indices)
        # # 到时候在这里写一个 select indices 然后进行选择
        if epoch > 1 and self.opt.with_selection:
            indices = self.select_smps(epoch)
        self.train_pair_like_i_1 = {k: v for k, v in self.train_pair_like_i.items()}

        batch = [[] for i in range(6)]
        batch_indices = []
        idx = 0
        while idx < len(indices):
            items = train_smp_pairs[indices[idx]]
            for i, item in enumerate(items):
                batch[i].append(item)
            batch_indices.append(indices[idx])
            if len(batch[0]) == self.opt.train_batchsize or idx + 1 == train_smp_pair_num:
                batch = self.dynamic_padding_train_batch(batch)
                self.train_batch_indices_cache = batch_indices
                yield batch
                batch = [[] for i in range(6)]
                batch_indices = []
            idx += 1


    def select_smps(self, epoch):
        remove_dup_idxs = set()
        indices = list(range(len(self.smps_rr)))
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
            random.shuffle(indices_selected)
            random.shuffle(review_indices)
            final_indices = [i for i in indices_selected + review_indices if i not in remove_dup_idxs]
        else:
            final_indices = [i for i in indices if i not in remove_dup_idxs]
            random.shuffle(final_indices)
        # remove_dup_idxs
        print(f'selected indices={len(indices_selected)}, review indices={len(review_indices)}')

        print(f'remove item {len([i for i in indices_selected if i in remove_dup_idxs])}')
        return final_indices

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

    def valid_or_test_batches(self, which='qr'):
        #the_smps = self.test_smps_qr if which == 'qr' else self.test_smp_pairs_qq
        the_smps = self.test_smp_pairs
        smp_pair_num = len(the_smps)
        batch = [[] for i in range(4)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                yield batch
                batch = [[] for i in range(4)]


if __name__ == '__main__':

    pass
    from logging import getLogger
    from KBQA.KBQA_WebQA.KBQA_webqa_glar_parameters import JointWebqaParameters

    KBQA_Webqa_DataManager(JointWebqaParameters(0), getLogger('t'))
    # pass
