# -*- coding: utf-8 -*-


import logging
import sys
import random
import torch
import torch.nn as nn
import layers
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from utils import AverageMeter

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x



class SQ_GLARModel(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, emb_vs, vocab_id2word,padding_idx=0, state_dict=None):
        super(SQ_GLARModel, self).__init__()


        self.opt = opt
        self.vocab_id2word = vocab_id2word

        self.logger = self.setup_loger()
        self.emb_dim = self.opt.emb_dim
        self.q_hidden = opt.q_hidden
        self.rel_hidden = opt.rel_hidden

        vocab_size = emb_vs.shape[0]
        assert self.opt.emb_dim == emb_vs.shape[1], print(emb_vs.shape)
        self.char_embedding = nn.Embedding(vocab_size, self.opt.emb_dim, padding_idx=padding_idx)
        self.char_embedding.weight.data = torch.FloatTensor(emb_vs)

        self._xor_match = layers._xor_match_knn(self.opt.knn_nums,self.vocab_id2word)

        self.flatten = layers.Flatten()
        self.dp_linear = nn.Dropout(opt.dropout_liner_output_rate)
        self.merge_fts_dim = opt.merge_fts_dim
        self.o_liner = nn.Linear(opt.merge_fts_dim * 2, 1, bias=False)  # opt.merge_fts_dim * 2
        self.o_liner_shallow = nn.Linear(opt.merge_fts_dim * 2, 1, bias=False)
        self.prob_out_linear = nn.Linear(2, 1, bias=False)

        self.drop_emb = nn.Dropout(self.opt.dropout_emb)
        self.rep_rnn_q = layers.StackedBRNN(
            input_size=self.emb_dim,
            hidden_size= self.q_hidden,
            num_layers=opt.sru_ly_num, #
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            # res_net=opt.res_net

        )

        self.rep_rnn_rel = layers.StackedBRNN(
            input_size= self.emb_dim,
            hidden_size=self.rel_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
        )

        self.q_rel_match = layers.Coattn(self.q_hidden * 2)

        match_in_dim = self.q_hidden * 2 * 3  # 2*3
        match_o_dim = 6 * self.q_hidden
        self.merge_liner = nn.Sequential(nn.Linear(match_in_dim, match_o_dim, bias=False),
                                         nn.Dropout(opt.dropout_liner_output_rate)
                                         )
        self.merge_liner2 = nn.Linear(match_o_dim + self.q_hidden * 2 * 2, opt.merge_fts_dim * 2, bias=False)
        self.merge_liner3 = nn.Linear(self.q_hidden * 2, opt.merge_fts_dim * 2, bias=False)
        self.merge_sru = layers.StackedBRNN(
            input_size=match_o_dim + self.q_hidden * 2 * 2,
            hidden_size=opt.merge_fts_dim,
            num_layers=1,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers
        )
        filter_nums = [50, 50, 50]
        self.merge_conv = layers.MyConv(emb_dim=match_o_dim + self.q_hidden * 2,
                                        filter_nums=filter_nums,
                                        window_sizes=[1, 3, 5],
                                        cnn_out_drop=opt.dropout_rnn
                                        )
        merge_fts_dim = 100
        self.o_liner = nn.Linear(opt.merge_fts_dim * 2, 1, bias=False)  # opt.merge_fts_dim * 2
        self.dual_liner = nn.Linear(2, 1, bias=False)
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.type_emb_trans_liner = nn.Linear(opt.emb_dim, self.q_hidden * 2, bias=False)

        if state_dict:
            new_state = set(self.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.load_state_dict(state_dict['network'])

        parameters = [p for p in self.parameters() if p.requires_grad]

        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        elif opt.optimizer == 'ada':
            self.optimizer = optim.Adadelta(parameters, lr=opt.learning_rate)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
        num_params = sum(p.data.numel() for p in parameters)
        self.logger.info("TraditionalRDModel, {} parameters".format(num_params))
        self.train_loss = AverageMeter()
        self.train_loss_not_null = AverageMeter()
        self.adv_train_loss = AverageMeter()

    def zero_loss(self):
        self.train_loss = AverageMeter()
        self.train_loss_not_null = AverageMeter()

    def forward(self, data, only_get_fts=False, adv_training=False, given_fts=False):
        """Inputs:
        """
        if given_fts:
            q_embs, rel_embs = data
        else:
            q_idxs, rel_idxs, q_knn_idxs, rel_knn_idxs = data
            # Embed both document and question
            q_embs = self.char_embedding(q_idxs)
            q_embs = self.drop_emb(q_embs)

            rel_embs = self.char_embedding(rel_idxs)

        lm_xor = self._xor_match(q_idxs, rel_idxs, q_knn_idxs, rel_knn_idxs) #print lm_xor
        lm_xor = lm_xor.type(torch.FloatTensor)

        h = lm_xor.size(2)#out_channels=32
        self.con1d = nn.Conv1d(in_channels=h, out_channels=32, kernel_size=2, stride=1)
        lm_conv = self.con1d(lm_xor.transpose(1, 2))
        lm_conv_dp = self.dp_linear(lm_conv)

        # (64,33,15)
        lm_conv_out = self.flatten(lm_conv_dp)

        # torch.Size([64, 384])
        self.shallow_linear = nn.Linear(lm_conv_out.size(-1), self.merge_fts_dim * 2, bias=False)
        lm_feat = self.shallow_linear(lm_conv_out)

        lm_feat = lm_feat.to('cuda')
        shallow_probs = self.o_liner_shallow(lm_feat)

        rel_embs = self.drop_emb(rel_embs)

        q_hidden_states = self.rep_rnn_q(q_embs)
        rel_hidden_states = self.rep_rnn_rel(rel_embs)

        if self.opt.without_interactive:
            rel_match_states = F.max_pool1d(rel_hidden_states.transpose(1, 2),
                                            kernel_size=rel_hidden_states.size(1)).squeeze(-1).unsqueeze(1).expand(
                q_hidden_states.size())
        else:
            rel_match_states, rel_predicate_match_states = self.q_rel_match(q_hidden_states, rel_hidden_states) #print rel_predicate_match_states

        merge_in = torch.cat([rel_match_states, q_hidden_states], dim=2)
        merge_in = torch.cat([merge_in, rel_match_states * q_hidden_states], dim=2)

        merge_in = self.merge_liner(merge_in)

        merge_in = F.relu(merge_in)
        merge_in = torch.cat([merge_in, rel_match_states, q_hidden_states], dim=2)
        merge_res = self.merge_liner2(merge_in)
        predicate = self.merge_liner3(rel_predicate_match_states)
        fts_q = F.max_pool1d(merge_res.transpose(1, 2), kernel_size=merge_res.size(1)).squeeze(-1)
        fts_p = F.max_pool1d(predicate.transpose(1, 2), kernel_size=predicate.size(1)).squeeze(-1)

        probs = self.cosine(fts_q, fts_p)
        probs = probs.view(shallow_probs.size())

        shallow_probs = F.sigmoid(shallow_probs)

        merge_probs = torch.cat([probs, shallow_probs], dim=1)

        out = self.prob_out_linear(merge_probs)

        if adv_training:
            return out, (q_embs, rel_embs)
        return out

    def update(self, batch):
        self.train()
        q_idxs, g_rel_idxs, cdt_rel_idxs, g_wh_sts, neg_wh_sts, g_knn_idxs, neg_knn_idxs, q_knn_idxs = batch
        if torch.cuda.is_available():
            q_idxs, g_rel_idxs, cdt_rel_idxs,g_knn_idxs, neg_knn_idxs,q_knn_idxs = [Variable(torch.from_numpy(e).long().cuda()) for e in
                                                (q_idxs, g_rel_idxs, cdt_rel_idxs, g_knn_idxs, neg_knn_idxs, q_knn_idxs)]

        else:
            q_idxs, g_rel_idxs, cdt_rel_idxs,g_knn_idxs, neg_knn_idxs, q_knn_idxs = [Variable(torch.from_numpy(e).long()) for e in
                                                (q_idxs, g_rel_idxs, cdt_rel_idxs, g_knn_idxs, neg_knn_idxs, q_knn_idxs)]

        g_score = self((q_idxs, g_rel_idxs, q_knn_idxs, g_knn_idxs))
        neg_score = self((q_idxs, cdt_rel_idxs, q_knn_idxs, neg_knn_idxs))

        loss = F.margin_ranking_loss(g_score, neg_score,
                                     target=Variable(torch.ones(g_score.size())).cuda(),
                                     margin=0.5)
        loss_flag = F.margin_ranking_loss(g_score, neg_score,
                                          target=Variable(torch.ones(g_score.size())).cuda(),
                                          margin=0.5, reduce=False) > 0
        not_null = loss_flag.byte().sum().item() / q_idxs.size(0)

        self.train_loss_not_null.update(not_null, len(q_idxs))
        self.train_loss.update(loss.data[0], len(q_idxs))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        self.optimizer.step()

        acc = (g_score - neg_score).gt(0).float().sum() / q_idxs.size(0)

        score_cost = 1 - (g_score - neg_score)
        score_cost = F.margin_ranking_loss(g_score, neg_score,
                                           target=Variable(torch.ones(g_score.size())).cuda(),
                                           margin=0.7, reduce=False)
        return acc.data[0], score_cost.view(-1).data.cpu().numpy()

    def adv_update(self, batch):
        self.train()
        q_idxs, g_rel_idxs, cdt_rel_idxs = batch
        if torch.cuda.is_available():
            q_idxs1, g_q_idxs2, neg_q_idxs2 = [Variable(torch.from_numpy(e).long().cuda()) for e in
                                               (q_idxs, g_rel_idxs, cdt_rel_idxs)]
        else:
            q_idxs1, g_q_idxs2, neg_q_idxs2 = [Variable(torch.from_numpy(e).long()) for e in
                                               (q_idxs, g_rel_idxs, cdt_rel_idxs)]
        g_score, fts1 = self((q_idxs1, g_q_idxs2), adv_training=True)
        neg_score, fts2 = self((q_idxs1, neg_q_idxs2), adv_training=True)

        if torch.cuda.is_available():
            loss = F.margin_ranking_loss(g_score, neg_score,
                                         target=Variable(torch.ones(g_score.size())).cuda(),
                                         margin=0.5)
        else:
            loss = F.margin_ranking_loss(g_score, neg_score,
                                         target=Variable(torch.ones(g_score.size())),
                                         margin=0.5)
        grad_fts1 = grad(loss, fts1, retain_graph=True)
        grad_fts2 = grad(loss, fts2, retain_graph=True)
        # adv_rate = 0.01
        # adv_rate = 0.05
        adv_rate = self.opt.adv_rate
        if torch.cuda.is_available():

            adv_g_score = self(
                [adv_rate * Variable(F.normalize(grad_true_fts.data)).cuda() + true_fts for true_fts, grad_true_fts
                 in zip(fts1, grad_fts1)], given_fts=True)
            adv_neg_score = self(
                [adv_rate * Variable(F.normalize(grad_true_fts.data)).cuda() + true_fts for true_fts, grad_true_fts
                 in zip(fts2, grad_fts2)], given_fts=True)
        else:
            adv_g_score = self(
                [adv_rate * Variable(F.normalize(grad_true_fts.data)) + true_fts for true_fts, grad_true_fts
                 in zip(fts1, grad_fts1)], given_fts=True)
            adv_neg_score = self(
                [adv_rate * Variable(F.normalize(grad_true_fts.data)) + true_fts for true_fts, grad_true_fts
                 in zip(fts2, grad_fts2)], given_fts=True)
        adv_g_score = F.sigmoid(adv_g_score)
        adv_neg_score = F.sigmoid(adv_neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        if torch.cuda.is_available():
            adv_loss = F.margin_ranking_loss(adv_g_score, adv_neg_score,
                                             target=Variable(torch.ones(adv_g_score.size())).cuda(),
                                             margin=0.5)
        else:
            adv_loss = F.margin_ranking_loss(adv_g_score, adv_neg_score,
                                             target=Variable(torch.ones(adv_g_score.size())),
                                             margin=0.5)
        self.train_loss.update(loss.data[0], q_idxs1.size(0))
        self.adv_train_loss.update(adv_loss.data[0], q_idxs1.size(0))

        total_loss = loss + adv_loss
        # optimizer = self.optimizer_qq if which == 'qq' else self.optimizer
        optimizer = self.optimizer
        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        optimizer.step()
        acc = (g_score - neg_score).gt(0).float().sum() / q_idxs1.size(0)
        if torch.cuda.is_available():
            score_cost = F.margin_ranking_loss(g_score, neg_score,
                                               target=Variable(torch.ones(g_score.size())).cuda(),
                                               margin=0.6, reduce=False)
        else:
            score_cost = F.margin_ranking_loss(g_score, neg_score,
                                               target=Variable(torch.ones(g_score.size())),
                                               margin=0.6, reduce=False)
        return acc.data[0], score_cost.view(-1).data.cpu().numpy()

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # 'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.logger.info('model saved to {}'.format(filename))
        except BaseException:
            self.logger.warning('[ WARN: Saving failed... continuing anyway. ]')

    def predict_score_of_batch(self, batch):
        with torch.no_grad():
            self.eval()
            if torch.cuda.is_available():
                q_idxs, rel_idxs,_, r_knn_idxs, q_knn_idxs = [Variable(torch.from_numpy(e).long(), volatile=True).cuda() for e in batch]

            else:
                q_idxs, rel_idxs, r_knn_idxs, q_knn_idxs = [Variable(torch.from_numpy(e).long(), volatile=True) for e in batch]

            score = self((q_idxs, rel_idxs, q_knn_idxs, r_knn_idxs))
            # score = F.sigmoid(score)
            score = score.view(-1)
            return score.data.cpu().numpy()

    def setup_loger(self):
        # setup logger
        log = logging.getLogger('TraditionalModel')
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
    pass
