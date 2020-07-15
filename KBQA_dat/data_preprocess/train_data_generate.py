# -*- coding: utf-8 -*-

import KBQA_dat.KBQA_file_paths as dt_p
import json
import re

def delte_punctuation_kbqa(sentence):
    """
    :param sentence:
    :return:
    """
    sentence = sentence.lower()  # lowercase
    if "who's" in sentence:
        sentence = re.sub("who's", "who is", sentence)
    if "what's" in sentence:
        sentence = re.sub("what's", "what is", sentence)
    if "where's" in sentence:
        sentence = re.sub("where's", "where is", sentence)
    if "which's" in sentence:
        sentence = re.sub("which's", "which is", sentence)
    if "how's" in sentence:
        sentence = re.sub("how's", "how is", sentence)

    sentence = re.sub('\'s', " ", sentence)
    sentence = re.sub("_", " ", sentence)
    # delete: ,.'":;?()[]

    sentence = re.sub(r'[\.\!\/,()\[\]:?;$%^*\\\"\']', " ", sentence)
    sentence = re.sub("[\s]+", " ", sentence.strip())
    return sentence



#self.rels = [ln.strip('\n') for ln in open(dt_p.web_qa_rel_f).readlines()]
def generate_relation_list():

    sq_train_negative_re = [ln_in.strip() for ln in open(dt_p.Yu_sq_train_negative_relations).readlines() for ln_in in ln.strip('\n').split(' ')]
    sq_valid_negative_re = [ln_in.strip() for ln in open(dt_p.Yu_sq_valid_negative_relations).readlines() for ln_in in ln.strip('\n').split(' ')]
    sq_test_negative_re  = [ln_in.strip() for ln in open(dt_p.Yu_sq_train_negative_relations).readlines() for ln_in in ln.strip('\n').split(' ')]

    sq_train_preprocess_re = [ln.strip('\n').split('\t')[1].strip() for ln in open(dt_p.Yu_sq_train_preprocess, encoding='utf-8').readlines()]
    sq_valid_preprocess_re = [ln.strip('\n').split('\t')[1].strip() for ln in open(dt_p.Yu_sq_valid_preprocess,encoding='utf-8').readlines()]
    sq_test_preprocess_re = [ln.strip('\n').split('\t')[1].strip() for ln in open(dt_p.Yu_sq_test_preprocess,encoding='utf-8').readlines()]

    with open(dt_p.Yu_sq_entity_link_tuples, "r", encoding="utf-8") as f:
        candidates_str = json.load(f)
    sq_test_link_re = [can.strip().split("==")[1].strip() for candidate in candidates_str for can in candidate]

    re_records = [sq_train_negative_re, sq_valid_negative_re, sq_test_negative_re,sq_train_preprocess_re
                  ,sq_valid_preprocess_re,sq_test_preprocess_re,sq_test_link_re]
    re_list = []
    for sq_re in re_records:
        for re in sq_re:
            if re not in re_list:
                re_list.append(re)
    print("The count of relations in train/dev/test data is %d" % len(re_list))

    re_file= open(dt_p.KBQA_SQ_re_list, 'w')
    for re in re_list:
        re_file.write(re)
        re_file.write('\n')
    re_file.close()

def read_re_list():
    id_to_rel = [ln.strip('\n') for ln in open(dt_p.KBQA_SQ_re_list).readlines()]
    rel2id_dict = {}
    for id in range(len(id_to_rel)):
        rel2id_dict[id_to_rel[id]] = id

    return rel2id_dict, id_to_rel

def read_entity_name_list():
    mid2_entity = {}
    entity2mid = {}
    train_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_train_entity_name,encoding="utf-8").readlines() for ln_pair
                  in ln.strip('\n').split('\t')]
    valid_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_valid_entity_name,encoding="utf-8").readlines() for ln_pair
                  in ln.strip('\n').split('\t')]
    test_mid_entity = [ln_pair for ln in open(dt_p.Yu_sq_test_entity_name, encoding="utf-8").readlines() for ln_pair
                        in ln.strip('\n').split('\t')]

    for i in range(0, len(test_mid_entity), 2):
        mid2_entity[test_mid_entity[i]] = test_mid_entity[i+1]
        entity2mid[test_mid_entity[i+1]] = test_mid_entity[i]
    return mid2_entity, entity2mid

def transform_data():
    rel2id, id2rel = read_re_list()
    mid2_entity, entity2mid = read_entity_name_list()

    sq_train_negative_re = [ln.split() for ln in open(dt_p.Yu_sq_train_negative_relations).readlines()]
    sq_valid_negative_re = [ln.split() for ln in open(dt_p.Yu_sq_valid_negative_relations).readlines()]
    sq_test_negative_re  = [ln.split() for ln in open(dt_p.Yu_sq_test_negative_relations).readlines()]

    sq_train_preprocess_re = [ln.strip() for ln in open(dt_p.Yu_sq_train_preprocess, encoding='utf-8').readlines()]
    sq_valid_preprocess_re = [ln.strip() for ln in open(dt_p.Yu_sq_valid_preprocess, encoding='utf-8').readlines()]
    sq_test_preprocess_re  = [ln.strip() for ln in open(dt_p.Yu_sq_test_preprocess, encoding='utf-8').readlines()]

    sq_preprocess_file = [sq_test_preprocess_re]
    negative_re_file = [sq_test_negative_re]
    save_files = [dt_p.KBQA_SQ_test]  # KBQA_SQ_valid


    for sq_preprocess_re_s, negative_re_s, save_file in zip(sq_preprocess_file, negative_re_file,save_files):
        new_prepocess_re_lines = []

        for prepocess_re_line, negative_res_line in zip(sq_preprocess_re_s,negative_re_s):
            negative_res = [rel2id.get(str(re)) for re in negative_res_line]
            negative_res_str = ' '.join(str(i) for i in negative_res)
            gold_re = rel2id.get(str(prepocess_re_line.split('\t')[1]))

            e_entity = mid2_entity[prepocess_re_line.split('\t')[0]]
            question_ne = prepocess_re_line.split('\t')[3].replace(mid2_entity[prepocess_re_line.split('\t')[0]]
                                                                   ,'#head_entity#')
            new_prepocess_re_lines.append('\t'.join([prepocess_re_line.strip(),str(gold_re).strip(),
                                               negative_res_str.strip(),question_ne.strip()]))


        re_file = open(save_file, 'w', encoding='utf-8')
        for line in new_prepocess_re_lines:
            re_file.write(line)
            re_file.write('\n')
        re_file.close()



if __name__ == '__main__':
    pass
    #generate_relation_list()
    #transform_data()

    #test
    q_fs = [u'sq_{}'.format(i) for i in 'train.replace_ne valid.replace_ne test'.split(" ")]
    print(q_fs)


