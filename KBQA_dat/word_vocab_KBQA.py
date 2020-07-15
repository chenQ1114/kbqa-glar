# -*- coding: utf-8 -*-

import numpy as np
import KBQA_dat.KBQA_file_paths as dt_p
import sys

class WordVocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""
    @staticmethod
    def load_glove(dim=300):
        f = open(dt_p.glove_dp+'glove.6B.{}d.txt'.format(dim))
        glove = {}
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) < dim + 1:
                continue
            else:
                glove[tokens[0]] = map(float, tokens[1:])
        return glove

    def __init__(self, vocab_file, emb_dim, with_glove=False):
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        self.UNKNOWN_TOKEN = '<UNK>'
        self.RELSEP = '#rel_sep#'
        self.RELPAD = '####relpad###'

        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim
        if with_glove:
            glove = self.load_glove()
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        self.vs = []
        num = 0
        self.vs_to_word = {}
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.RELSEP, self.RELPAD]:   
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            if with_glove and w in glove:  
                self.vs.append(glove[w])
                num += 1
            else:
                if w == self.PAD_TOKEN:
                    self.vs.append(0 * np.random.uniform(-0.5, 0.5, self.emb_dim))

                else:
                    self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))

        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                lns = [ln.strip('\n') for ln in vocab_f.readlines()]

        else:
            with open(vocab_file, 'r') as vocab_f:
                lns = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]


        for ln in lns:
            w, vs = ln.split('\t')
            vs = [float(v) for v in vs.split(',')]
            if len(w) == 0 or len(vs) != emb_dim:
                continue
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.array(vs))
            self.vs_to_word[tuple(vs)] = w

        assert len(self.vs) == len(self._word_to_id)
        self.vs = np.asarray(self.vs)

        print(u"WordVocab: Finished constructing vocabulary of {} total words.".format(self._count))


    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]



if __name__ == '__main__':
    # vocab = WordVocab(dt_p.KBQA_SQ_vocab, emb_dim=300)
    vocab = WordVocab(dt_p.KBQA_Web_vocab, emb_dim=300)
    print(vocab.word2id('sports/sports_facility/home_venue_for..sports/team_venue_relationship/from'))
