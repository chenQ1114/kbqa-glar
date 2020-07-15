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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(dev)

import numpy as np
import torch

import sys
import logging
import json
import os
from GLARSimpleQA.SimpleQAParapmeters import RDParameters as JointParameters
from GLARSimpleQA.sq_DataManager import GLARSimpleQADataManager
from utils import AverageMeter
from GLARSimpleQA.sq_glar_model import SQ_GLARModel as SQ_Model
from KNNdata.SimpleQAKNNData import SimpleQAKNNDataManager

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def Normalization(x):
    return [(float(i) - np.min(x)) / (np.max(x) - np.min(x)) for i in x]


def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x) / s, lst)

class NerTest2:
    def __init__(self, opt=JointParameters(train_idx='none', dev=dev)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt = GLARSimpleQADataManager(self.opt, self.loger)
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

        self.kn = SimpleQAKNNDataManager()
        self.kn.read_knn_list()

    def adjust_learning_rate(self, epoch):
        optimizer = self.model.optimizer
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.75 ** int(epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def test(self, epoch):
        self.loger.info('================================= epoch: {} =================================='.format(epoch))
        self.loger.info('loss={}'.format(self.model.train_loss.avg))

        self.loger.info(f'train_loss_not_null={self.model.train_loss_not_null.sum}, '
                        f'total_train_num={self.model.train_loss_not_null.count},')

        valid_or_test='test'
        print("in test",valid_or_test)

        scores = self.predict_all_batches(self.dt.valid_or_test_batches(valid_or_test), 'r')
        rel_test_acc,relation_scores = self.infer_acc_rel_v_test(scores,
                                               self.dt.cdt_num_of_test_dat)
        self.loger.info(u'{}, r_f1={}'.format(valid_or_test, rel_test_acc))

        for q, gold_name, rel_name, test_score in zip(self.dt.question_text, self.dt.gold_name_of_test, self.dt.relation_text, relation_scores):
            get_answer = True
            for score in test_score:
                if score > test_score[0]:
                    get_answer = False
            if not get_answer:
                print(str(q) +' |||' + str(gold_name))
                print(rel_name)
                print(test_score)


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

            if max(q_scores[:g_num]) > max(q_scores[g_num:]): 
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
            if max(q_scores[:g_num]) > max(q_scores[g_num:]):  
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
        # model = KBQA_SQ_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict)  # 重启的时候
        model = SQ_Model(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict
                              ,vocab_id2word=self.dt.vocab._id_to_word)
        # model.embedding_ly.weight.requires_grad = False
        return model


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


# TestGLARModel
if __name__ == '__main__':

    opt = JointParameters(dev=dev, train_idx = 'sq_glar', epoch = 40)


    t = NerTest2(opt=opt)

    t.test(opt.epoch_out)
