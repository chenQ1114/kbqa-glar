# -*- coding: utf-8 -*-

import os
import pickle

dp = os.path.abspath(os.path.dirname(__file__)) + '/'


# ###########################################  KBQA data #############################################

KBQA_SQ_dp = dp + u'KBQA_re_data/KBQA_sq_relations/'
KBQA_SQ_re_list = KBQA_SQ_dp + u'relation.2M.list'
KBQA_SQ_train = KBQA_SQ_dp + u'sq_train.replace_ne'
KBQA_SQ_valid = KBQA_SQ_dp + u'sq_valid.replace_ne'
KBQA_SQ_test = KBQA_SQ_dp + u'sq_test'

KBQA_SQ_entity_link_tuples = KBQA_SQ_dp + u'sq_test_entity_link_tuples.json'

glove_dp = dp + u'glove.6B/'
glove_300 = glove_dp + u'glove.6B.300d.txt'

KBQA_vocab_dp = dp + u'KBQA_re_data/KBQA_vocab/'
KBQA_SQ_vocab = KBQA_vocab_dp + u'sq_vocab.txt'

KBQA_KNN_dp = dp +u'KNN/'
k8_kbqa_sq_f_0 = KBQA_KNN_dp + u'kbqa_sq_knn_8_0.txt'
k8_kbqa_sq_f_1 = KBQA_KNN_dp + u'kbqa_sq_knn_8_1.txt'
k8_kbqa_sq_f = KBQA_KNN_dp + u'kbqa_sq_knn_8.txt'

# ###########################################  coling2018GNNQA true whole data#############################################
KBQA_Web_coling_true_whole_dp = dp + u'KBQA_re_data/KBQA_webqa_true_whole_coling_GNNQA/'
KBQA_Web_coling_true_whole_re_list = KBQA_Web_coling_true_whole_dp + u'relations.txt'
KBQA_Web_coling_test_entity_linking_true_whole_update = KBQA_Web_coling_true_whole_dp + u'coling_test_true_whole_entity_linking_update.txt'
KBQA_Web_coling_true_whole_train = KBQA_Web_coling_true_whole_dp  + u'GnnQA.RE.true_whole.train.txt'
KBQA_Web_coling_true_whole_test = KBQA_Web_coling_true_whole_dp  + u'GnnQA.RE.true_whole.test.txt'

KBQA_Web_coling_ture_whole_vocab = KBQA_vocab_dp + u'kbqa_webqa_GnnQA_true_whole_vocab.txt'
kbqa_web_coling_true_whole_knn_f = KBQA_KNN_dp + u'kbqa_web_GnnQA_true_whole_knn.txt'





if __name__ == '__main__':
    pass
