# -*- coding: utf-8 -*-

import sys
import logging
import json
import os
import math
import time



dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

dev = 1

import numpy as np
import torch

import sys
import logging
import json
import os

from KBQA_SQ.KBQA_sq_DataManager import KBQA_SQ_DataManager
from KBQA_SQ.KBQA_sq_Parameters import KBQA_SQParameters as JointParameters

from KBQA_SQ.kbqa_sq_glar_model import KBQA_SQ_GLARModel as KBQA_SQ_Model

from utils import AverageMeter

dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

sys.path.append('../..')
from KBQA_dat.data_preprocess.train_data_generate import delte_punctuation_kbqa
from KNNdata.KBQA_sq_KNNData import KBQA_SQ_KNNDataManager

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def Normalization(x):
    return [(float(i)-np.min(x))/(np.max(x)-np.min(x)) for i in x]

def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)

def longestCommonSubstring_score(question, sub_name, weight):
    question = delte_punctuation_kbqa(question)
    ques_list = question.strip().split(" ")
    sub_name = sub_name.lower()
    sub_list = sub_name.strip().split(" ")

    ques_len = len(ques_list)
    sub_len = len(sub_list)

    record = [[0 for i in range(sub_len+1)] for j in range(ques_len+1)]

    maxNum = 0
    p = 0

    for i in range(ques_len):
        for j in range(sub_len):
            if ques_list[i] == sub_list[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i+1

    score = weight * (float(maxNum)/float(ques_len)) + (1-weight) * (float(maxNum)/float(sub_len))
    return score


class NerTest:
    def __init__(self, opt=JointParameters(train_idx='none', dev=dev)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt = KBQA_SQ_DataManager(self.opt, self.loger)
        # self.train()

        self.model = self.resume()

        self.model.cuda()
        self.loger.info(json.dumps(dict(self.opt), indent=True))
        self.best_valid_f1 = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.error_in_traindat = 0

        self.fuc_substring_weight = 0.6
        self.substring_weight = 0.2

        self.kn = KBQA_SQ_KNNDataManager()
        self.kn.read_knn_list()

    def adjust_learning_rate(self, epoch):
        optimizer = self.model.optimizer
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.75 ** int(epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def valid_it(self, epoch, batch_num):
        self.loger.info('================================= epoch: {} =================================='.format(epoch))
        self.loger.info('loss={}'.format(self.model.train_loss.avg))

        self.loger.info(f'train_loss_not_null={self.model.train_loss_not_null.sum}, '
                        f'total_train_num={self.model.train_loss_not_null.count},')
        valid_or_test='valid'
        scores = self.predict_all_batches(self.dt.valid_or_test_batches(valid_or_test), 'r')
        rel_acc = self.infer_acc_rel_v(scores, self.dt.cdt_num_of_valid_dat if valid_or_test == 'valid' else self.dt.cdt_num_of_test_dat)

        if rel_acc > self.best_valid_f1 and valid_or_test == 'valid':
            self.loger.info('new best valid f1 found')
            self.best_valid_f1 = rel_acc
            if rel_acc > 0.938:
                    fnm = self.opt.model_dir + 'model_epoch_{}.h5.'.format(epoch)
                    self.model.save(fnm, epoch)
            self.loger.info(u'{}, r_f1={}, batch_num={}, need'.format(valid_or_test, rel_acc, batch_num))

        valid_or_test='test'
        print("in test",valid_or_test)

        scores = self.predict_all_batches(self.dt.valid_or_test_batches(valid_or_test), 'r')
        rel_test_acc,relation_scores = self.infer_acc_rel_v_test(scores,
                                               self.dt.cdt_num_of_test_dat)

        self.loger.info(u'{}, r_f1={}, batch_num={}, need'.format(valid_or_test, rel_test_acc, batch_num))


    def kbqa_test_simple_ranking_score_add_commonsubstring_save_samesubject(self):
        test_questions_subject_mids_relations = self.dt.test_questions_subject_mids_relations
        candidates_mid_relation_midscore_objects = self.dt.candidates_mid_relation_midscore_objects
        total = len(test_questions_subject_mids_relations)
        right_0_0 = 0
        right_rels = 0

        mid2_entity, entity2mid = self.dt.read_entity_name_list()

        assert len(test_questions_subject_mids_relations) == len(candidates_mid_relation_midscore_objects)
        max_relation_more_than_one = 0  # æœ‰å¤šä¸ªè°“è¯­å¾—åˆ†æœ€é«˜çš„æƒ…å†µç»Ÿè®¡
        smps = []
        for q_idx, (gold_relations, candidate_mid_relation, q, gr, cdt_rs) in enumerate(
                zip(test_questions_subject_mids_relations,
                    candidates_mid_relation_midscore_objects, self.dt.test_qs, self.dt.test_gold_rels,
                    self.dt.test_cdt_rels)):
            question_str, gold_mid, gold_relation = gold_relations
            print(question_str + "||" + mid2_entity[gold_mid] + "||" + self.dt.rels[int(gold_relation)])
            if len(candidate_mid_relation) == 0:
                continue
            elif len(candidate_mid_relation) == 1:
                print("candidate_relation_with_1" + question_str + "||" + mid2_entity[gold_mid] + "||" + self.dt.rels[
                    int(gold_relation)])
                right_0_0 += 1
                right_rels += 1
                continue
            smp = []
            assert len(candidate_mid_relation) == len(cdt_rs)
            print_candi_mid = []
            print_candi_rels = []
            q_x = self.dt.rep_of_question(delte_punctuation_kbqa(q))
            q_knn_sts = [self.kn.knns[idx].tolist() for idx in q_x]
            for candi_relation, r_idx in zip(candidate_mid_relation, cdt_rs):
                # q_withen = delte_punctuation_kbqa(q).replace(candi_relation[5], '#head_entity#')
                print_candi_mid.append(candi_relation[5])
                rx1 = self.dt.idxs_of_rel(r_idx)
                rx_knn_sts = [self.kn.knns[idx].tolist() for idx in rx1]
                print_candi_rels.append(candi_relation[1])
                smp.append((q_x, rx1, rx_knn_sts, q_knn_sts))
            batch = [[] for i in range(4)]
            for idx, smp_pair in enumerate(smp):
                for i, item in enumerate(smp_pair):
                    batch[i].append(item)

            batch = self.dt.dynamic_padding_valid_batch(batch)
            candi_rela_scores_old = self.model.predict_score_of_batch(batch)
            candi_rela_scores = list(candi_rela_scores_old.reshape(-1))
            nor_candi_rela_scores = Normalization(candi_rela_scores)
            candi_mid_scores = []
            for candi_relation in candidate_mid_relation:
                mid_name = candi_relation[5]
                candi_mid_scores.append(longestCommonSubstring_score(q, mid_name, self.fuc_substring_weight))
            assert len(candi_rela_scores) == len(candi_mid_scores)
            candi_scores = [(a + b) for a, b in zip(nor_candi_rela_scores, candi_mid_scores)]

            max_candi_scores = max(candi_scores)
            candidate_scores_with_mid_relation = [[i, j] for i, j in zip(candi_scores, candidate_mid_relation)]
            print_candi_relation_with_mid_relation = [[i, j, k, l, m] for i, j, k, l, m in
                                                      zip(print_candi_mid, print_candi_rels,
                                                          candi_mid_scores, nor_candi_rela_scores,
                                                          candi_scores)]

            max_candidate_scores_with_mid_relation = [i for i in candidate_scores_with_mid_relation if
                                                      i[0] == max_candi_scores]

            if len(max_candidate_scores_with_mid_relation) > 1:
                max_relation_more_than_one += 1
            # print(max_candidate_scores_with_mid_relation[:3])
            sorted_candidate_scores_with_mid_relation = sorted(candidate_scores_with_mid_relation, key=lambda x: x[0],
                                                               reverse=True)

            sorted_print_candi_relation_with_mid_relation = sorted(print_candi_relation_with_mid_relation,
                                                                   key=lambda x: x[4],
                                                                   reverse=True)

            top_1_candidate = sorted_candidate_scores_with_mid_relation[0]

            flag_right = 0
            flag_rels = 0
            for top_candidate in sorted_candidate_scores_with_mid_relation[
                                 :len(max_candidate_scores_with_mid_relation)]:
                if mid2_entity[gold_mid] == top_candidate[1][5] and top_1_candidate[1][5] \
                        == top_candidate[1][5] and self.dt.rels[int(gold_relation)] == top_candidate[1][
                    1] and flag_right == 0:
                    right_0_0 += 1
                    flag_right = 1
                if self.dt.rels[int(gold_relation)] == top_candidate[1][1] and top_1_candidate[1][5] \
                        == top_candidate[1][5] and flag_rels == 0:
                    right_rels += 1
                    flag_rels = 1

            '''
            if gold_mid == top_1_candidate[1][0] and self.dt.rels[int(gold_relation)] == top_1_candidate[1][1]:
                right_0_0  += 1
            if self.dt.rels[int(gold_relation)] == top_1_candidate[1][1]:
                right_rels += 1

            flag_right = 0
            flag_rels = 0
            for top_candidate in sorted_candidate_scores_with_mid_relation[:len(max_candidate_scores_with_mid_relation)]:
                if gold_mid == top_candidate[1][0] and self.dt.rels[int(gold_relation)] == top_candidate[1][1] and flag_right==0:
                    right_0_0 += 1
                    flag_right = 1
                if self.dt.rels[int(gold_relation)] == top_candidate[1][1] and flag_rels==0:
                    right_rels +=1
                    flag_rels = 1           
            '''

        print("max_relation_more_than_one:", max_relation_more_than_one)
        print("right:", right_0_0)
        print("total:", total)
        print("right rels", right_rels)
        print("acc:", float(right_0_0) / float(total))

        return right_0_0, total, float(right_0_0) / float(total)


    def predict_all_batches(self, batches, which):
        s = time.time()
        all_scores = []
        for batch in batches:
            scores = self.model.predict_score_of_batch(batch)
            all_scores.append(scores)
        all_scores = np.concatenate(all_scores, axis=0)
        e = time.time()
        # print(e-s, 'inference time')

        return [x for x in list(all_scores.reshape(-1))]

    def infer_acc_rel_v_test(self, scores, cdt_num_of_valid_dat):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0
        relation_scores = []
        for num in cdt_num_of_valid_dat:
            g_num = 1
            if num == 1:
                relation_scores.append(scores[pos: pos + num])
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]
            relation_scores.append(q_scores)

            if max(q_scores[:g_num]) > max(q_scores[g_num:]):  # 而且这里用的是大于 很充沛了
                acc_num += 1
            pos += num
        assert pos == len(scores)
        return acc_num / len(cdt_num_of_valid_dat), relation_scores

    def infer_acc_rel_v(self, scores, cdt_num_of_valid_dat):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0
        for num in cdt_num_of_valid_dat:
            g_num = 1
            if num == 1:
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]
            # if len(q_scores) != len(set(q_scores)):
            #     print(q_scores, 'hey there is dup values')
            # print(q_scores)
            if max(q_scores[:g_num]) > max(q_scores[g_num:]):  # 而且这里用的是大于 很充沛了
                acc_num += 1
            pos += num
        assert pos == len(scores)
        # print(acc_num, len(cdt_num_of_valid_dat))
        return acc_num / len(cdt_num_of_valid_dat)

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

    #
    def resume(self):
        self.loger.info('[loading previous model...]')
        checkpoint = torch.load(self.opt.trained_model)
        opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        self.loger.info(json.dumps(dict(opt), indent=True))
        #model = KBQA_SQ_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict)  # 重启的时候
        model = KBQA_SQ_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict,
                              vocab_id2word = self.dt.vocab._id_to_word)
        # model.embedding_ly.weight.requires_grad = False
        return model


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

#TestGLARModel
if __name__ == '__main__':
    opt = JointParameters(dev=dev, train_idx='kbqa_sq_glar')


    t = NerTest(opt=opt)
    # t.valid_it(-1,0)


    t.kbqa_test_simple_ranking_score_add_commonsubstring_save_samesubject()

