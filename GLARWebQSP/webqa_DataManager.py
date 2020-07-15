# -*- coding: utf-8 -*-

from collections import defaultdict
import random
import os
import sys
from tqdm import tqdm
import numpy as np
import logging
import word_vocab
import dat.file_pathes as dt_p
from KNNdata.WebQAKNNData import WebQAKNNDataManager


class GLARWebqaDataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.vocab = word_vocab.WordVocab(dt_p.webqa_vocab_f, emb_dim=300)


        self.kn = WebQAKNNDataManager(k = self.opt.knn_nums)
        self.kn.read_knn_list(k = self.opt.knn_nums)
        self.k_value = self.kn.k

        q_fs = [u'{}WebQSP.RE.{}.with_boundary.withpool.dlnlp.txt'.format(dt_p.web_q_f_dp, i) for i in
                'train test'.split()]
        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.web_qa_rel_f).readlines()]
        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {} 
        self.train_pair_like_i = {}

        self.rel_questions = self.get_query_of_rel()

        self.test_smps_qr, self.cdt_num_of_test_dat, self.gold_rel_num_test_dat, self.groups_of_rels_cnt, \
        self.cdt_rels_of_q = self.a_f2x_valid_and_test(q_fs[1])
        self.question_text, self.relation_text, self.relation_num_dat, self.relation_id_dat = self.a_f2x_test_text(
            q_fs[1])
        self.gold_relation_text = self.a_f2x_text_gold_rels(q_fs[1])
        self.smps_rr = self.a_f2x(self.q_fs[0])
        self.ever_contribute = set()

    def get_query_of_rel(self):
        rel_questions = defaultdict(list)

        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        for r, qs in rel_questions.items():
            rel_questions[r] = list(set(qs))
        return rel_questions

    def calculate_idf(self):
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        qs = [q for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs))]
        w_df = defaultdict(int)
        for q in qs:
            for w in q.split():
                w_df[w] += 1
        w_idf = {w: np.log(len(qs) / (4 + df)) for w, df in w_df.items()}
        return w_idf

    def rank_alias_qs(self, rel_idx, questions):
        rel_ws = set(self.words_of_web_rel(rel_idx))
        vs = []
        questions = list(set(questions))
        for q in questions:
            q_ws = q.split()
            v = sum([1 for w in q_ws if w in rel_ws])
            v += 1e-4 * len(q_ws)
            vs.append(v)
        indexes = sorted(range(len(vs)), key=lambda k: vs[k])
        sorted_qs = [questions[idx] for idx in indexes]
        return sorted_qs

    # ######################################### trans to idx ################################################

    def words_of_web_rel(self, rel_idx):
        """
        """
        rel = self.rels[rel_idx]
        ws = []
        rels = []
        for sub_rel in rel.split('..'):
            tmp_sub_rel = '/' + sub_rel.replace('.', '/')
            rels.append(tmp_sub_rel)
            ws += self.words_of_a_rel(tmp_sub_rel)

        while len(rels) < 2:
            rels = [self.vocab.RELPAD] + rels
        if len(rels) > 2:
            rels = rels[:2]

        return ws + rels

    def words_of_a_rel(self, rel):
        """
        """
        return rel.split('/')[-1].split('_')


    def idxs_of_rel(self, rel_idx):
        """

        :param rel_idx: 
        :return:
        """
        return self.vocab.seqword2id(self.words_of_web_rel(rel_idx))

    def question_of_the_rel_random(self, rel, tmp_question=None):
        questions = self.rel_questions.get(rel, [])
        if len(questions) == 0:
            return ''
        if len(questions) == 1 and tmp_question == questions[0]:
            return ''
        q = random.choice(questions)
        while q == tmp_question:
            q = random.choice(questions)
        return q

    def all_question_of_the_rel(self, rel, tmp_question=None):
        questions = self.rel_questions.get(rel, [])
        return questions

    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())

    def a_f2x_valid_and_test(self, f):
        """
        :param f:
        :return:
        """
        gold_rel, cdt_rels, qs = self.parse_f(f)
        smps_qr = []

        cdt_num_of_valid_dat = []
        gold_rel_num_q = []
        groups_of_rels_cnt = []
        test_gold_rels = []
        cdt_rels_of_q = []

        all_null_rels = []
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            grs = (gr if isinstance(gr, list) else [gr])
            test_gold_rels.append(grs)
            neg_cdts = [r for r in set(cdt_rs) if r not in grs]
            neg_cdts = sorted(neg_cdts)
            q_cdt_rels = grs + neg_cdts
            flags = ['g'] * len(grs) + ['n'] * len(neg_cdts)
            q_x = self.rep_of_question(q)
            q_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in q_x]

            groups = []
            g_rels_q_num = 0
            cdt_q_num = 0
            tmp_cdt_rels = []
            assert len(q_cdt_rels) == len(flags)
            null_rels = []
            for r, flag in zip(q_cdt_rels, flags):

                rx1 = self.idxs_of_rel(r)
                rx1_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in rx1]
                smps_qr.append((q_x, rx1, rx1_knn_valid_test_sts, q_knn_valid_test_sts))
                questions = self.rel_questions.get(r, [])[:self.opt.valid_q_max_num]
                if len(questions) == 0:
                    null_rels.append(r)
                if flag == 'g':
                    g_rels_q_num += len(questions)
                cdt_q_num += len(questions)
                groups.append(len(questions))
                tmp_cdt_rels.append(r)
            cdt_rels_of_q.append(tmp_cdt_rels)
            groups_of_rels_cnt.append(groups)
            cdt_num_of_valid_dat.append(cdt_q_num)
            gold_rel_num_q.append(g_rels_q_num)
            all_null_rels.append(null_rels)
        self.test_gold_rels = test_gold_rels
        self.all_null_rels = all_null_rels

        self.test_cdt_num_of_rels = [len(cdts) for cdts in cdt_rels_of_q]
        self.test_gold_num_of_rels = [len(cdts) for cdts in test_gold_rels]
        return smps_qr, cdt_num_of_valid_dat, gold_rel_num_q, groups_of_rels_cnt, cdt_rels_of_q

    def a_f2x_test_text(self,f):
        gold_rel, cdt_rels, qs = self.parse_f(f)

        cdt_num_of_valid_dat = []
        cdt_name_of_valid_dat = []
        cdt_id_of_valid_dat = []

        test_gold_rels=[]
        question = []
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
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
        test_gold_rels=[]
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            grs = (gr if isinstance(gr, list) else [gr])
            test_gold_rels.append(grs)
            q_cdt_rels = grs

            q_cdt_relname = [self.rels[r] for r in q_cdt_rels]

            gold_name_of_test.append(q_cdt_relname)
        return gold_name_of_test

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f, silent=True)
        smps_rr = []
        self.get_train = True
        gt_1 = 0

        more_than_one_golds = {}
       
        for q_idx, (grs, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)
            q_knn_sts = [self.kn.knns[idx].tolist() for idx in q_x]
            if not isinstance(grs, list):
                grs = [grs]
            dup_q = defaultdict(list)
            for gr in grs:
                gr_x = self.idxs_of_rel(gr)
                gr_x_wh = self.rel_wh_word_sts.get(gr, [0] * len(self.whs_idxs))
                grx_knn_sts = [self.kn.knns[idx].tolist() for idx in gr_x]
                for r in cdt_rs:
                    if r in grs:
                        continue
                    smps_rr_idx = len(smps_rr)
                    dup_q[gr].append(smps_rr_idx)
                    r_x = self.idxs_of_rel(r)
                    rx_knn_sts = [self.kn.knns[idx].tolist() for idx in r_x]
                    r_x_wh = self.rel_wh_word_sts.get(r, [0] * len(self.whs_idxs))
                    smps_rr.append((q_x, gr_x, r_x, grx_knn_sts, rx_knn_sts,q_knn_sts))
            if len(grs) > 1:
                gt_1 += 1
            more_than_one_golds[q_idx] = dup_q
        self.get_train = False
        self.more_than_one_golds = more_than_one_golds
        return smps_rr

    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [0] * (max_len - len(idxs))
            # idxs_batch_padded.append(idxs + to_add)
            idxs_batch_padded.append(to_add + idxs) 
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
        :param batch:
        :return:
        """

        q_idxs, g_alias_q_idxs, neg_alias_q_idxs, g_knn_sts,negknn_sts, qknn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_alias_q_idxs = self.dynamic_padding(g_alias_q_idxs)
        neg_alias_q_idxs = self.dynamic_padding(neg_alias_q_idxs)

        g_knn_sts = self.dynamic_padding_for_knn_idxs(g_knn_sts)
        negknn_sts = self.dynamic_padding_for_knn_idxs(negknn_sts)
        qknn_sts = self.dynamic_padding_for_knn_idxs(qknn_sts)

        batch = q_idxs, g_alias_q_idxs, neg_alias_q_idxs, g_knn_sts, negknn_sts, qknn_sts
        return list(np.array(i) for i in batch)

    def max_alias_len(self, g_rel_alias_q_idxs):
        vs = []
        for q_idxs in g_rel_alias_q_idxs:
            vs.append(max([len(q) for q in q_idxs]))
        return max(vs)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, r_knn_sts, q_knn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        r_knn_sts = self.dynamic_padding_for_knn_idxs(r_knn_sts)
        q_knn_sts = self.dynamic_padding_for_knn_idxs(q_knn_sts)

        batch = q_idxs, cdt_rel_idxs, r_knn_sts, q_knn_sts
        return list(np.array(i) for i in batch)

    def get_train_batchs(self, epoch):
        self.smps_rr = self.a_f2x(self.q_fs[0])
        train_smp_pairs = self.smps_rr
        train_smp_pair_num = len(train_smp_pairs)
        indices = list(range(train_smp_pair_num))

        random.shuffle(indices)
        if epoch > 1 and self.opt.with_selection:
            indices = self.select_smps(epoch)
        self.train_pair_like_i_1 = {k: v for k, v in self.train_pair_like_i.items()}

        batch = [[] for i in range(8)]
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
                batch = [[] for i in range(8)]
                batch_indices = []
            idx += 1

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

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


        indices_selected = last_indices
        indices_selected = list(indices_selected)
        review_indices = random.sample(need_review_indices, k=int(self.opt.review_rate * len(need_review_indices)))

        assert len(indices_selected) == len(set(indices_selected))

        if epoch > 10:
            random.shuffle(indices_selected)
            random.shuffle(review_indices)
            final_indices = [i for i in indices_selected + review_indices if i not in remove_dup_idxs]
        else:
            final_indices = [i for i in indices if i not in remove_dup_idxs]
            random.shuffle(final_indices)

        print(f'selected indices={len(indices_selected)}, review indices={len(review_indices)}')
        print(f'remove item {len([i for i in indices_selected if i in remove_dup_idxs])}')
        return final_indices

    def valid_or_test_batches(self, which='qr'):
      
        the_smps = self.test_smps_qr
        smp_pair_num = len(the_smps)
        batch = [[] for i in range(6)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                yield batch
                batch = [[] for i in range(6)]

    @staticmethod
    def parse_f(fnm, silent=False):
        """
        :param fnm:
        :return:
        """
        lns = [ln.strip('\n') for ln in open(fnm).readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []
        if not silent:
            print(len(lns), fnm)
        for ln in lns:
            gr, cdt_rs, q = ln.split('\t')
            qs.append(q)
            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r) - 1) 
                gold_rel.append(multi_gold_r)
            else:
                gold_rel.append(int(gr) - 1)

            if cdt_rs == 'noNegativeAnswer':
                cdt_rels.append([])
            else:
                cdt_rels.append([int(i) - 1 for i in cdt_rs.split()])
        return gold_rel, cdt_rels, qs




if __name__ == '__main__':
    from logging import getLogger
    from GLARWebQSP.webqa_parameters import JointWebqaParameters
    GLARWebqaDataManager(JointWebqaParameters(0), getLogger('t'))
    pass
