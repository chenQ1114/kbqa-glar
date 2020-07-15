# -*- encoding: utf-8 -*-

from collections import defaultdict
import random
import os
import sys
from tqdm import tqdm
import numpy as np
import logging
import word_vocab
import dat.file_pathes as dt_p
from KNNdata.SimpleQAKNNData import SimpleQAKNNDataManager

from GLARSimpleQA.SimpleQAParapmeters import RDParameters

class GLARSimpleQADataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.alias_q_max_len = self.opt.alias_q_max_len

        self.vocab = word_vocab.WordVocab(dt_p.simple_qa_vocab_f, emb_dim=300)
        
        self.kn = SimpleQAKNNDataManager(k = self.opt.knn_nums)
        self.kn.read_knn_list(k = self.opt.knn_nums)
        self.k_value = self.kn.k

        q_fs = [u'{}{}.replace_ne.withpool'.format(dt_p.simple_qa_dp, i) for i in 'valid train test'.split()]
        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.simple_qa_rel_f).readlines()]
        self.rel_questions = self.get_query_of_rel()
        self.rel_wh_word = self.get_query_wh_of_rel()
        self.rel_wh_word_sts = self.get_query_wh_of_rel_more()
        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {}  # 上一个epoch的结果
        self.train_pair_like_i = {}
        self.parsed_train_dat_cache = self.parse_f(q_fs[1])

        self.valid_smp_pairs, self.cdt_num_of_valid_dat = self.a_f2x_valid_and_test(q_fs[0])
        self.test_smp_pairs, self.cdt_num_of_test_dat = self.a_f2x_valid_and_test(q_fs[2])
        self.question_text, self.relation_text, self.relation_num_dat, self.relation_id_dat = self.a_f2x_test_text(q_fs[2])
        self.gold_name_of_test = self.a_f2x_test_gold_answer_text(q_fs[2])
        self.get_train = True
        self.train_smp_pairs = self.a_f2x(self.q_fs[1])
        self.get_train = False

    def get_query_wh_of_rel(self):
        rel_questions = defaultdict(list)

        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        rel_questions_wh = dict()

        whs = 'when where who what'.split()

        for r, qs in rel_questions.items():
            whs_cnt = defaultdict(int)
            for q in qs:
                for wh in whs:
                    if wh in set(q.split()):
                        whs_cnt[wh] += 1
            max_wh = max(whs, key=lambda wh: whs_cnt[wh])
            rel_questions_wh[r] = max_wh  
            rel_name = self.rels[r]

        return rel_questions_wh

    def get_query_wh_of_rel_more(self):
        rel_questions = defaultdict(list)
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        rel_questions_wh_sts = dict()

        whs = 'when where who what'.split()
        self.whs_idxs = self.vocab.seqword2id(whs)
        for r, qs in rel_questions.items():
            whs_cnt = [0] * len(whs)
            for q in qs:
                for wh_idx, wh in enumerate(whs):
                    if wh in set(q.split()):
                        whs_cnt[wh_idx] += 1
            whs_cnt = np.asarray(whs_cnt) / len(qs) # sum(whs_cnt) + 0.00001) # len(qs)  #
            rel_questions_wh_sts[r] = whs_cnt  
            rel_name = self.rels[r]

        return rel_questions_wh_sts

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

    # ######################################### trans to idx ################################################
    @staticmethod
    def words_of_rel(rel):

        return u'_'.join([rel.split('/')[-2]]).split('_') + rel.split('/')[-1].split('_')
      

    def idxs_of_rel(self, rel_idx):

        r_ws = self.words_of_rel(self.rels[rel_idx])

        rel_x = self.vocab.seqword2id(r_ws) + [self.vocab.word2id(self.rels[rel_idx])]

        if not self.opt.use_wh_hot:
            return rel_x

        wh = self.rel_wh_word.get(rel_idx, 'nu')   

        if self.get_train and random.randint(1, 100) > 10:
            wh = 'nu'
        rel_x = [self.vocab.word2id(wh)] + rel_x   
        return rel_x

    def question_of_the_rel(self, rel, tmp_question):
        cdts = self.rel_questions.get(rel, [None])
        question = random.choice(cdts) # + [None])
        while question == tmp_question:
            if len(set(cdts)) > 1:
                question = random.choice(cdts )# + [None])
            else:
                return ''
        if question is None:
            return ''
        return question

    def rep_of_alias_questions(self, rel, tmp_question, for_valid=False):
        if for_valid:
            rel_questions = self.rel_questions.get(rel, [])[:self.opt.alias_num]
            while len(rel_questions) < self.opt.alias_num:
                rel_questions.append('')
        else:
            rel_questions = [self.question_of_the_rel(rel, tmp_question) for i in range(self.opt.alias_num)]
        rep_of_questions = [self.rep_of_question(q) for q in rel_questions]
        return rep_of_questions

    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parsed_train_dat_cache
        smps = []
        for gr, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)  #word2id
            q_knn_sts = [self.kn.knns[idx].tolist() for idx in q_x]
            gr_x = self.idxs_of_rel(gr)

            gr_text = self.rels[gr] 

            # gr_alias_q_x = self.rep_of_alias_questions(gr, q)
            grx_wh_sts = self.rel_wh_word_sts.get(gr, [0] * len(self.whs_idxs))
            grx_knn_sts = [self.kn.knns[idx].tolist() for idx in gr_x]

            for r in cdt_rs:
                if r == gr:
                    continue
                r_x = self.idxs_of_rel(r)
                rx_wh_sts = self.rel_wh_word_sts.get(r, [0] * len(self.whs_idxs))
                rx_knn_sts = [self.kn.knns[idx].tolist() for idx in r_x]

                smps.append((q_x, gr_x, r_x, grx_wh_sts, rx_wh_sts, grx_knn_sts, rx_knn_sts,q_knn_sts))

        return smps
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

    def a_f2x_test_gold_answer_text(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f)
        cdt_num_of_valid_dat = []
        gold_name_of_test = []
        cdt_id_of_valid_dat = []
        question = []
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            q_gold_rels = [gr]
            q_gold_relname = [self.rels[r] for r in q_gold_rels]

            question.append(q)
            gold_name_of_test.append(q_gold_relname)
        return gold_name_of_test

    def a_f2x_valid_and_test(self, f):

        gold_rel, cdt_rels, qs = self.parse_f(f)
        smps = []
        cdt_num_of_valid_dat = []

        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            q_x = self.rep_of_question(q)
            q_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in q_x]
            q_cdt_rels = [gr] + [r for r in set(cdt_rs) if r != gr]

            cdt_num_of_valid_dat.append(len(q_cdt_rels))

            for r_idx in q_cdt_rels:
                rx1 = self.idxs_of_rel(r_idx)
                rx1_knn_valid_test_sts = [self.kn.knns[idx].tolist() for idx in rx1]
                rx_wh_sts = self.rel_wh_word_sts.get(r_idx, [0] * len(self.whs_idxs))
                smps.append((q_x, rx1, rx_wh_sts,rx1_knn_valid_test_sts,q_knn_valid_test_sts))

        return smps, cdt_num_of_valid_dat

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

        q_idxs, g_rel_idxs, cdt_rel_idxs, gwh_sts, negwh_sts, g_knn_sts,negknn_sts, qknn_sts = batch

        q_idxs = self.dynamic_padding(q_idxs)
        g_rel_idxs = self.dynamic_padding(g_rel_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)


        g_knn_sts = self.dynamic_padding_for_knn_idxs(g_knn_sts)
        negknn_sts = self.dynamic_padding_for_knn_idxs(negknn_sts)
        qknn_sts = self.dynamic_padding_for_knn_idxs(qknn_sts)


        batch = q_idxs, g_rel_idxs, cdt_rel_idxs, gwh_sts, negwh_sts, g_knn_sts, negknn_sts, qknn_sts

        return list(np.array(i) for i in batch)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, wh_sts, r_knn_sts, q_knn_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        r_knn_sts = self.dynamic_padding_for_knn_idxs(r_knn_sts)
        q_knn_sts = self.dynamic_padding_for_knn_idxs(q_knn_sts)

        batch = q_idxs, cdt_rel_idxs, wh_sts, r_knn_sts, q_knn_sts  
        return list(np.array(i) for i in batch)

    def get_train_batchs(self, epoch, for_keras=False):
        self.train_smp_pair_num = len(self.train_smp_pairs)
        print(self.train_smp_pair_num, 'training sample num')
        # Q1, true_p1, neg_p1
        # Q1, true_p1, neg_p2
        # Q2, true_p1, neg_p1...

        indices = list(range(self.train_smp_pair_num))
        # if epoch < 1:
        random.shuffle(indices) #shuffle
        #batch = [[] for i in range(5)]
        batch = [[] for i in range(8)]
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
                batch = [[] for i in range(8)]
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

        indices_selected = last_indices
        indices_selected = list(indices_selected)

        review_indices = random.sample(need_review_indices, k=int(self.opt.review_rate * len(need_review_indices)))

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
        batch = [[] for i in range(5)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                if for_keras:
                    batch[2] = batch[2].reshape(batch[2].shape[0], batch[2].shape[1] * batch[2].shape[2])
                yield batch
                #batch = [[] for i in range(4)]
                batch = [[] for i in range(5)]

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

    @staticmethod
    def parse_f(fnm):

        lns = [ln.strip('\n') for ln in open(fnm, encoding='utf-8').readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []
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

class TestTrain:
    def __init__(self, opt=RDParameters(train_idx='none', dev=1)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt_dual = GLARSimpleQADataManager(self.opt, self.loger)
    def setup_loger(self):
        # setup logger
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.opt.log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        log.addHandler(fh)
        log.addHandler(ch)
        return log

if __name__ == '__main__':
    opt = RDParameters(dev=1, train_idx='JustTest')

    t = TestTrain(opt=opt)
    pass
