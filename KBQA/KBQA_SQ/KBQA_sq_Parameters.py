# -*- coding: utf-8 -*-

class KBQA_SQParameters:
    def __init__(self, train_idx, dev, epoch = 39):
        # self.train_batchsize = 256
        #self.train_batchsize = 256
        self.train_batchsize = 64
        self.valid_batchsize = 512

        # self.alias_num = 3
        self.alias_num = 1
        self.alias_q_max_len = 10

        self.match_o_chen_dim=300
        self.emb_dim = 300

        self.q_hidden = 150
        self.rel_hidden = self.q_hidden
        self.rel_alias_hidden = self.q_hidden
        self.match_o_dim = 6 * self.q_hidden

        self.merge_fts_dim = 250

        self.merge_kernel_sizes = [1, 2, 2]
        self.merge_filter_nums = [50, 50, 150]

        self.sru_ly_num = 1

        self.dropout_emb = 0.35
        self.dropout_rnn = 0.1
        self.dropout_rnn_output = True
        self.dropout_liner_output_rate = 0.25
        self.concat_rnn_layers = False
        self.res_net = False

        self.with_selection = True
        # self.with_selection = False
        self.with_wh = True
        self.with_only_dynamic_wh = False
        self.use_wh_hot = False

        self.without_interactive = False
        self.optimizer = 'adamax'
        self.learning_rate = 0.002
        self.adv_rate = 0.00025
        self.weight_decay = 0
        self.review_rate = 0.35
        # self.momentum = 0
        self.train_idx = train_idx
        self.dev = dev
        self.log_file = 'logs/Hyper_parameters/{}.log'.format(train_idx)
        self.model_dir = 'models/Hyper_parameters/{}/'.format(train_idx)
        self.drop_wh_sts = 10
        # self.resume_dsrc_flag = False  False when it is training
        self.resume_dsrc_flag = False #True when it is test for KBQA

        self.max_q_len = 20
        self.max_p_len = 10

        #self.knn_nums = 8
        self.knn_nums = 8
        #self.epoch_out = 39
        self.epoch_out = epoch
        self.trained_model =self.model_dir + 'model_epoch_{}.h5.'.format(self.epoch_out)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


if __name__ == '__main__':
    pass

