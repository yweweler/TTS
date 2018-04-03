import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - query: (batch, 1, dim) or (batch, dim)
            - annots: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)

        # (batch, max_time, 1)
        alignment = self.v(nn.functional.tanh(
            processed_query + processed_annots))

        # (batch, max_time)
        return alignment.squeeze(-1)


class AttentionRNN(nn.Module):
    def __init__(self, hidden_dim, annot_dim, memory_dim):
        super(AttentionRNN, self).__init__()
        self.rnn_cell = nn.GRUCell(hidden_dim + memory_dim, hidden_dim)
        self.alignment_model = BahdanauAttention(
            annot_dim, hidden_dim, hidden_dim)

    def forward(self, memory, context, rnn_state, annotations):
        # Concat input query and previous context context
        rnn_input = torch.cat((memory, context), -1)
        #rnn_input = rnn_input.unsqueeze(1)

        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i}, s_{i-1})
        rnn_output = self.rnn_cell(rnn_input, rnn_state)

        # Alignment
        # (batch, max_time)
        # e_{ij} = a(s_{i-1}, h_j)
        alignment = self.alignment_model(annotations, rnn_output)

        # Normalize context weight
        alignment = F.softmax(alignment, dim=-1)

        # Attention context vector
        # (batch, 1, dim)
        # c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
        context = torch.bmm(alignment.unsqueeze(1), annotations)
        context = context.squeeze(1)
        return rnn_output, context, alignment


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, annot_dim, query_dim):
        super(AttentionLayer, self).__init__()
        self.alignment_model = BahdanauAttention(
            annot_dim, query_dim, hidden_dim)

    def forward(self, annotations, query):
        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i}, s_{i-1})
        # s = self.query_proj(query)

        # Alignment
        # (batch, max_time)
        # e_{ij} = a(s_{i-1}, h_j)
        alignment = self.alignment_model(annotations, query)

        # Normalize context weight
        alignment = F.softmax(alignment, dim=-1)

        # Attention context vector
        # (batch, 1, dim)
        # c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
        context = torch.bmm(alignment.unsqueeze(1), annotations)
        context = context.squeeze(1)
        return context, alignment
