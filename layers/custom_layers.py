# coding: utf-8
import torch
from torch import nn


class ZoneOutCell(nn.Module):

    def __init__(self, cell, zoneout_prob=0):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob

    def forward(self, inputs, hidden):
        def zoneout(h, next_h, prob):
            if isinstance(h, tuple):
                num_h = len(h)
                if not isinstance(prob, tuple):
                    prob = tuple([prob] * num_h)
                return tuple([zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])
            mask = h.new(h.size()).bernoulli_(prob)
            return mask * next_h + (1 - mask) * h

        next_hidden, next_cell = self.cell(inputs, hidden)
        if hidden[0].shape[1] ==  next_hidden.shape[1]:
            next_hidden = zoneout(hidden[0], next_hidden, self.zoneout_prob)
        return (next_hidden, next_cell)