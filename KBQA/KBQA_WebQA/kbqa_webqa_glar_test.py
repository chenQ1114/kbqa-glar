import collections
import random
import sys
import logging
import json
import os
import math
import time

dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

dev = 0


import numpy as np
import torch

import sys
import logging
import json
import os

from KBQA_WebQA.KBQA_webqa_glar_parameters import JointWebqaParameters
from KBQA_WebQA.KBQA_webqa_DataManager import KBQA_Webqa_DataManager
from KBQA_WebQA.kbqa_webqa_glar_model import KBQA_Webqa_GLARModel as KBQA_WEBQA_Model

from utils import AverageMeter

dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

sys.path.append('../..')
import KBQA_dat.KBQA_file_paths as dt_p
from KBQA_dat.data_preprocess.train_data_generate import delte_punctuation_kbqa
from KNNdata.KBQA_webqa_KNNData import KBQA_WEBQA_KNNDataManager

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
    def __init__(self, opt=JointWebqaParameters(train_idx='none', dev=dev)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt = KBQA_Webqa_DataManager(self.opt, self.loger)
        self.model = self.resume()
        self.model.cuda()
        self.loger.info(json.dumps(dict(self.opt), indent=True))
        self.best_valid_f1 = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.error_in_traindat = 0

        self.fuc_substring_weight = 0.6 #default = 0.6
        self.substring_weight = 0.2

        self.kn = KBQA_WEBQA_KNNDataManager(vocab_path=dt_p.KBQA_Web_coling_ture_whole_vocab, train_save_path= dt_p.kbqa_web_coling_true_whole_knn_f,k = self.opt.knn_nums)
        self.test_knn_list = self.kn.read_knn_list(k = self.opt.knn_nums)

    def adjust_learning_rate(self, epoch):
        optimizer = self.model.optimizer
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.75 ** int(epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

    def infer_acc_rel_v(self, scores, cdt_num_of_valid_dat, gold_res_num_of_valid_dat):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        assert len(cdt_num_of_valid_dat)==len(self.dt.candidate_cdt_rels)
        acc_num = 0
        pos = 0
        for num, g_num, candi_rel in zip(cdt_num_of_valid_dat, gold_res_num_of_valid_dat,self.dt.candidate_cdt_rels):
            if num == 1:
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]
            # print(q_scores)
            # print(candi_rel)
            if max(q_scores[:g_num]) > max(q_scores[g_num:]):
                acc_num += 1
            pos += num
        assert pos == len(scores)
        # print(acc_num, len(cdt_num_of_valid_dat))
        return acc_num / len(cdt_num_of_valid_dat)


    def check_error2(self, gold_path):
        gold_rel, cdt_rels, qs = self.parse_f(gold_path)

        candidates_mid_relation_midscore_objects = self.dt.candidates_mid_relation_midscore_objects
        candidates_mid_rels = self.dt.candidate_cdt_rels
        test_num = 10
        total = test_num
        cdt_num_of_valid_dat = []
        right_0_0=0
        for q_idx, (gr, candi_rels, q) in enumerate(zip(gold_rel[:10], candidates_mid_rels[:10], qs[:10])):
            if len(candi_rels)==0 or len(candi_rels)==1:
                right_0_0 = right_0_0 +1
                continue
            smps = []
            gold_rel_len=0
            cdt_rels_len =0
            q_x = self.dt.rep_of_question(q)
            q_knn_valid_test_sts = [self.test_knn_list[idx].tolist() for idx in q_x]

            grs = (gr if isinstance(gr, list) else [gr])
            gold_rel_len= len(grs)
            neg_cdts = [r for r in candi_rels if r not in grs]
            # neg_cdts = sorted(neg_cdts)
            q_cdt_rels = grs + neg_cdts

            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            tmp_cdt_rels = []

            for r_idx in q_cdt_rels:
                rx1 = self.dt.idxs_of_rel(r_idx)
                rx1_knn_valid_test_sts = [self.test_knn_list[idx].tolist() for idx in rx1]
                tmp_cdt_rels.append(r_idx)

                smps.append((q_x, rx1,rx1_knn_valid_test_sts,q_knn_valid_test_sts))
            cdt_rels_len = len(tmp_cdt_rels)

            if cdt_rels_len== gold_rel_len:
                right_0_0 = right_0_0+1
                continue

            batch = [[] for i in range(4)]
            for idx, smp_pair in enumerate(smps):
                for i, item in enumerate(smp_pair):
                    batch[i].append(item)
            batch = self.dt.dynamic_padding_valid_batch(batch)
            candi_rela_scores_old = self.model.predict_score_of_batch(batch)
            candi_rela_scores = list(candi_rela_scores_old.reshape(-1))
            nor_candi_rela_scores = Normalization(candi_rela_scores)

            result=[]
            print('result:')
            for score,rel in zip(nor_candi_rela_scores, candi_rels):
                result.append((self.dt.rels[int(rel)],rel, score))
            print(result)

            if max(nor_candi_rela_scores[:gold_rel_len]) > max(nor_candi_rela_scores[gold_rel_len:]):
                right_0_0 += 1
        print("right:", right_0_0)
        print("total:", total)
        print("acc:", float(right_0_0) / float(total))

    def check_error_in_test(self):

        right_0_0 = 0
        right_rels = 0
        total_result_dict = []

        gold_qs = self.dt.test_qs
        gold_subjects = self.dt.test_gold_subjects
        gold_rels = self.dt.test_gold_rels
        gold_indexs = self.dt.test_q_indexs
        gold_objects = self.dt.test_objects

        total = len(gold_indexs)

        candidates_mid_relation_midscore_objects = self.dt.candidates_mid_relation_midscore_objects
        candidates_mid_rels = self.dt.candidate_cdt_rels
        assert len(gold_qs)==len(gold_rels)==len(gold_indexs)==len(candidates_mid_relation_midscore_objects)==len(candidates_mid_rels)

        print_q_index = []

        for id,(gold_q, gold_s, gold_rel, gold_index, gold_o, candi_example, candi_rel) in enumerate(zip(gold_qs,gold_subjects,gold_rels,gold_indexs,gold_objects,candidates_mid_relation_midscore_objects,candidates_mid_rels)):
            if len(candi_example) == 0:
                continue
            if len(candi_example) == 1:
                right_0_0 += 1
                right_rels += 1
                output = collections.OrderedDict()
                output["QuestionId"] = 'WebQTest-'+str(gold_index)
                output["Answers"] = candi_example[0][2]
                total_result_dict.append(output)
                continue
            gold_rel = (gold_rel if isinstance(gold_rel, list) else [gold_rel])
            gold_subject = (gold_s if isinstance(gold_s, list) else [gold_s])

            gold_object_set =[]
            for object in gold_o:
                for m in object:
                    if m not in gold_object_set:
                        gold_object_set.append(m)

            if len(candi_example) == len(gold_rel):
                right_0_0 += 1
                right_rels += 1
                output = collections.OrderedDict()
                output["QuestionId"] = 'WebQTest-'+str(gold_index)
                output["Answers"] = gold_object_set
                total_result_dict.append(output)
                continue
            assert len(candi_example)==len(candi_rel)
            smp = []
            q_x = self.dt.rep_of_question(gold_q)
            q_knn_sts = [self.test_knn_list[idx].tolist() for idx in q_x]
            g_triple = []

            for candi_relation, r_idx in zip(candi_example, candi_rel):
                if r_idx in gold_rel and candi_relation[0] in gold_subject:
                    g_triple.append(candi_relation)
                rx1 = self.dt.idxs_of_rel(int(r_idx))
                rx_knn_sts = [self.test_knn_list[idx].tolist() for idx in rx1]
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
            for candi_relation in candi_example:
                mid_name = str(candi_relation[3])
                candi_mid_scores.append(longestCommonSubstring_score(gold_q, mid_name, self.fuc_substring_weight))
            assert len(candi_rela_scores) == len(candi_mid_scores)
            candi_scores = [(a + b) for a, b in zip(nor_candi_rela_scores, candi_mid_scores)]

            max_candi_scores = max(candi_scores)
            candidate_scores_with_mid_relation = [[i, j] for i, j in zip(candi_scores, candi_example)]
            sorted_candidate_scores_with_mid_relation = sorted(candidate_scores_with_mid_relation, key=lambda x: x[0],
                                                               reverse=True)
            max_candidate_scores_with_mid_relation = [i for i in candidate_scores_with_mid_relation if
                                                      i[0] == max_candi_scores]

            pred_objects=[]
            top_candidate = sorted_candidate_scores_with_mid_relation[0]

            for m in top_candidate[1][2]:
                if m not in pred_objects:
                    pred_objects.append(m)
                if len(pred_objects) >= len(gold_object_set):
                    break


            if top_candidate[1] in g_triple:
                right_0_0 += 1
            else:
                print_q_index.append(gold_index)

            output = collections.OrderedDict()
            output["QuestionId"] = 'WebQTest-' + str(gold_index)
            output["Answers"] = pred_objects
            total_result_dict.append(output)

        print("right:", right_0_0)
        print("total:", total)
        print("right rels", right_rels)
        print("acc:", float(right_0_0) / float(total))


        return right_0_0, float(right_0_0) / float(total), total_result_dict



    def kbqa_test_simple_ranking_score_add_commonsubstring_save_samesubject(self):
        right_0_0 = 0
        right_rels = 0
        total_result_dict = []

        gold_qs = self.dt.test_qs
        gold_subjects = self.dt.test_gold_subjects
        gold_rels = self.dt.test_gold_rels
        gold_indexs = self.dt.test_q_indexs
        gold_objects = self.dt.test_objects

        total = len(gold_indexs)

        candidates_mid_relation_midscore_objects = self.dt.candidates_mid_relation_midscore_objects
        candidates_mid_rels = self.dt.candidate_cdt_rels
        assert len(gold_qs) == len(gold_rels) == len(gold_indexs) == len(
            candidates_mid_relation_midscore_objects) == len(candidates_mid_rels)

        for id, (gold_q, gold_s, gold_rel, gold_index, gold_o, candi_example, candi_rel) in enumerate(
                zip(gold_qs, gold_subjects, gold_rels, gold_indexs, gold_objects,
                    candidates_mid_relation_midscore_objects, candidates_mid_rels)):
            if len(candi_example) == 0:
                continue
            if len(candi_example) == 1:
                right_0_0 += 1
                right_rels += 1
                output = collections.OrderedDict()
                output["QuestionId"] = 'WebQTest-' + str(gold_index)
                output["Answers"] = candi_example[0][2]
                total_result_dict.append(output)
                continue
            gold_rel = (gold_rel if isinstance(gold_rel, list) else [gold_rel])
            gold_subject = (gold_s if isinstance(gold_s, list) else [gold_s])

            gold_object_set = []
            for object in gold_o:
                for m in object:
                    if m not in gold_object_set:
                        gold_object_set.append(m)

            if len(candi_example) == len(gold_rel):
                right_0_0 += 1
                right_rels += 1
                output = collections.OrderedDict()
                output["QuestionId"] = 'WebQTest-' + str(gold_index)
                output["Answers"] = gold_object_set
                total_result_dict.append(output)
                continue
            assert len(candi_example) == len(candi_rel)
            smp = []
            q_x = self.dt.rep_of_question(gold_q)
            q_knn_sts = [self.test_knn_list[idx].tolist() for idx in q_x]
            g_triple = []

            for candi_relation, r_idx in zip(candi_example, candi_rel):
                if r_idx in gold_rel and candi_relation[0] in gold_subject:
                    g_triple.append(candi_relation)
                rx1 = self.dt.idxs_of_rel(int(r_idx))
                rx_knn_sts = [self.test_knn_list[idx].tolist() for idx in rx1]
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
            for candi_relation in candi_example:
                mid_name = str(candi_relation[3])
                candi_mid_scores.append(longestCommonSubstring_score(gold_q, mid_name, self.fuc_substring_weight))
            assert len(candi_rela_scores) == len(candi_mid_scores)
            candi_scores = [(a + b) for a, b in zip(nor_candi_rela_scores, candi_mid_scores)]

            max_candi_scores = max(candi_scores)
            candidate_scores_with_mid_relation = [[i, j] for i, j in zip(candi_scores, candi_example)]
            sorted_candidate_scores_with_mid_relation = sorted(candidate_scores_with_mid_relation, key=lambda x: x[0],
                                                               reverse=True)
            max_candidate_scores_with_mid_relation = [i for i in candidate_scores_with_mid_relation if
                                                      i[0] == max_candi_scores]

            pred_objects = []
            if len(max_candidate_scores_with_mid_relation) > 1:
                triple_flag = 0
                for top_candidate in sorted_candidate_scores_with_mid_relation[
                                     :len(max_candidate_scores_with_mid_relation)]:
                    if top_candidate[1] in g_triple:
                        for m in top_candidate[1][2]:
                            if m not in pred_objects:
                                pred_objects.append(m)

                    if top_candidate[1] in g_triple and triple_flag==0:
                        right_0_0 += 1
                        triple_flag = 1

                if triple_flag==0:
                    pred_objects = []
                    for m in sorted_candidate_scores_with_mid_relation[0][1][2]:
                        if m not in pred_objects:
                            pred_objects.append(m)
                        if len(pred_objects) >= len(gold_object_set):
                            break
            else:

                top_candidate = sorted_candidate_scores_with_mid_relation[0]
                if top_candidate[1] in g_triple:
                    right_0_0 += 1

                for m in top_candidate[1][2]:
                    if m not in pred_objects:
                        pred_objects.append(m)
                    if len(pred_objects) >= len(gold_object_set):
                        break


            output = collections.OrderedDict()
            output["QuestionId"] = 'WebQTest-' + str(gold_index)
            output["Answers"] = pred_objects
            total_result_dict.append(output)

        print("right:", right_0_0)
        print("total:", total)
        print("right rels", right_rels)
        print("acc:", float(right_0_0) / float(total))

        return right_0_0, float(right_0_0) / float(total), total_result_dict



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

    def resume(self):
        self.loger.info('[loading previous model...]')
        checkpoint = torch.load(self.opt.trained_model)
        opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        self.loger.info(json.dumps(dict(opt), indent=True))
        #model = KBQA_SQ_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict)  # 重启的时候
        model = KBQA_WEBQA_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict,
                              vocab_id2word = self.dt.vocab._id_to_word)
        # model.embedding_ly.weight.requires_grad = False
        return model

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def calculate_f1(predAnswers, test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as load_f:
        goldData = json.load(load_f)

    PredAnswersById = {}
    for item in predAnswers:
        PredAnswersById[item["QuestionId"]] = item["Answers"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    numCorrect = 0
    for entry in goldData["Questions"]:
        skip = True
        for pidx in range(0, len(entry["Parses"])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"][
                "ParseQuality"] == "Complete":
                skip = False

        if (len(entry["Parses"]) == 0 or skip):
            continue

        total += 1

        id = entry["QuestionId"]

        if id not in PredAnswersById:
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        if len(entry["Parses"]) == 0:
            print("Empty parses in the gold set. Breaking!!")
            break

        predAnswers = PredAnswersById[id]

        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        for pidx in range(0, len(entry["Parses"])):
            pidxAnswers = entry["Parses"][pidx]["Answers"]
            prec, rec, f1 = CalculatePRF1(pidxAnswers, predAnswers)
            # prec, rec, f1 = computeF1(pidxAnswers, predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        if bestf1 == 1.0:
            numCorrect += 1

    print("Number of questions:", int(total))
    print("Average precision over questions: %.3f" % (precSum / total))
    print("Average recall over questions: %.3f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.3f" % (f1sum / total))
    print("F1 of average recall and average precision: %.3f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print("True accuracy (ratio of questions answered exactly correctly): %.3f" % (numCorrect / total))


def FindInList(entry, elist):
    for item in elist:
        if entry == item:
            return True
    return False

def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0,
                    1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0,
                    0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList) == 0:
        return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist = [x["AnswerArgument"] for x in goldAnswerList]
        plist = predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry, plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]

#TestGLARModel
if __name__ == '__main__':
    # opt = JointWebqaParameters(dev=dev, train_idx='kbqa_webqa_glar',epoch=89)


    t = NerTest(opt=opt)
    # test_way_1: just consider top1
    right_0_0, acc, total_result_dict = t.check_error_in_test()
    calculate_f1(total_result_dict, dt_p.KBQA_coling_test_original)


