# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)

class Word2VecHidden(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, hidden_size=100, padding_idx=0):
        super(Word2VecHidden, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.iW = nn.Parameter(FT(hidden_size, embedding_size).uniform_(-0.5, 0.5))
        self.oW = nn.Parameter(FT(hidden_size, embedding_size).uniform_(-0.5, 0.5))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return t.matmul(self.ivectors(v), t.transpose(self.iW, 1, 0))

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return t.matmul(self.ovectors(v), t.transpose(self.oW, 1, 0))


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None, tie_weights=False, fake_indices=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.tie_weights = tie_weights
        if weights is not None and fake_indices is not None:
            is_fake = t.zeros(4000).type(t.bool)
            is_fake[t.LongTensor(list(fake_indices))] = True
            # adjust weights here and zero them out
            self.weights_real = self.weights.detach().clone()
            self.weights_real[is_fake] = 0.0
            self.weights_fake = self.weights.detach().clone()
            self.weights_fake[~is_fake] = 0.0
            self.fake_indices = t.LongTensor(list(fake_indices))

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.fake_indices is None:
            if self.weights is not None:
                nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
            else:
                nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        else:
            if self.weights is not None:
                import ipdb;ipdb.set_trace()
                # do broadcasting to check the values
                is_fake = iword.view(-1, 1).eq(self.fake_indices).sum(1)
                n_fake = is_fake.sum()
                n_real = batch_size - n_fake
                # two times sampling
                nwords_fake = t.multinomial(self.weights_fake, n_fake * context_size * self.n_negs, replacement=True).view(n_fake, -1)
                nwords_real = t.multinomial(self.weights_real, n_real * context_size * self.n_negs, replacement=True).view(n_real, -1)
                # create empty tensor and use is_fake to assign the sampled words to it
                nwords = t.zeros(batch_size, context_size * self.n_negs).type(t.int)
                nwords[is_fake] = nwords_fake
                nwords[n_real] = nwords_real
            else:
                raise NotImplementedError()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        if self.tie_weights:
            ovectors = self.embedding.forward_i(owords)
            nvectors = self.embedding.forward_i(nwords).neg()
        else:
            ovectors = self.embedding.forward_o(owords)
            nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()
