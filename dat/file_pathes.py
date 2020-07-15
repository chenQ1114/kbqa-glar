# -*- encoding: utf-8 -*-

import os
import pickle

dp = os.path.abspath(os.path.dirname(__file__)) + '/'

common_used_static_log_f = dp + 'common_used_static.log'

# ########################################### SimpleQA #############################################
simple_qa_dp = dp + u'KBQA_RE_data/sq_relations/'
simple_qa_rel_f = simple_qa_dp + u'relation.2M.list'

simple_qa_vocab_f = dp + 'SimpleQA_vocab.txt'

# ########################################### WebQA #############################################
web_q_f_dp = dp + u'KBQA_RE_data/webqsp_relations/'
web_qa_rel_f = web_q_f_dp + 'relations.txt'

webqa_vocab_f = dp + 'WebQA_vocab.txt'
k8_webqa_f = dp + u'KNN/webqa_knn_8.txt'

# ########################################### query log data #####################################
query_log_f = dp + 'all_query_res_sp.txt'

knn_simple_f = dp + u'KNN/sq_knn.txt'
k8_simple_f = dp + u'KNN/sq_knn_8.txt'
#k_simple_f_edit_distance = dp + u'KNN/sq_knn_edit_distance.txt'
if __name__ == '__main__':
    pass
