# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
from TTS.utils.text.symbols import symbols
from TTS.layers.attention import AttentionLayer


class Conv1dBlock(nn.Module):
    """
        Args:
            in_features (int): sample size
            K (int): max filter size in conv bank
            projections (list): conv channel sizes for conv projections
            num_highways (int): number of highways layers

        Shapes:
            - input: batch x time x dim
            - output: batch x time x dim*2
    """

    def __init__(self, in_features, out_features, hidden_features, kernel_size, dilation):
        super(Conv1dBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.padding = int((kernel_size) // 2)
        self.dil_padding = int(((kernel_size - 1) * self.dilation) // 2)

        self.relu = nn.ReLU()
        # TODO: Try no dilation for conv1
        self.conv1 = nn.Conv1d(in_features, hidden_features, self.kernel_size,
                               dilation=self.dilation, padding=self.dil_padding)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.conv2 = nn.Conv1d(hidden_features, out_features, self.kernel_size,
                               dilation=self.dilation, padding=self.dil_padding)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.scale = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        x_ = self.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        assert x.shape[2] == x_.shape[2], " >  ! Size mismatch {} vs {}".format(
            x.shape, x_.shape)
        if self.in_features != self.out_features:
            x = self.scale(x)
        x_ = self.relu(x_ + x)
        return x


class Conv1dBank(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, dilations, kernel_sizes):
        super(Conv1dBank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_sizes = kernel_sizes
        self.hidden_features = hidden_features
        self.blocks = []
        for i, dil in enumerate(dilations):
            kernel_size = kernel_sizes[i]
            if i+1 == len(dilations):
                if len(dilations) == 1:
                    self.blocks.append(Conv1dBlock(
                        in_features, out_features, self.hidden_features, kernel_size=kernel_size, dilation=dil))
                else:
                    self.blocks.append(Conv1dBlock(
                        hidden_features, out_features, self.hidden_features, kernel_size=kernel_size, dilation=dil))
            elif i == 0:
                self.blocks.append(Conv1dBlock(in_features, self.hidden_features,
                                               self.hidden_features, kernel_size=kernel_size, dilation=dil))
            else:
                self.blocks.append(Conv1dBlock(self.hidden_features, self.hidden_features,
                                               self.hidden_features, kernel_size=kernel_size, dilation=dil))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        # Needed to perform conv1d on time-axis
        # (B, in_features, T_in)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return x.contiguous()


class EncoderConv(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_dim, hidden_dim):
        super(EncoderConv, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim,
                                      max_norm=1.0)
        self.conv_bank = Conv1dBank(embed_dim, out_dim, hidden_dim,
                                    kernel_sizes=[5, 3, 3, 3], dilations=[1, 2, 4, 8])

    def forward(self, inputs):
        r"""
        Args:
            inputs (FloatTensor): embedding features

        Shapes:
            - inputs: batch x time x in_features
            - outputs: batch x time x 128*2
        """
        out = self.embedding(inputs)
        out = self.conv_bank(out)
        return out


class DecoderConvWithBuffer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, buffer_len=17, memory_dim=80):
        super(DecoderConvWithBuffer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.buffer_dim = in_dim + memory_dim
        self.buffer_len = buffer_len
        self.buffer_size = buffer_len * self.buffer_dim
        self.memory_dim = memory_dim

        self.buffer_proj_linear = self._get_linear(
            self.buffer_size, self.buffer_dim)
        self.memory_proj_linear = self._get_linear(out_dim, memory_dim)
        self.attention = AttentionLayer(hidden_dim, in_dim, self.buffer_size)
        self.conv_layer = Conv1dBank(self.buffer_dim, hidden_dim, hidden_dim,
                                     kernel_sizes=[5, 3, 3], dilations=[1, 2, 4])  # Receptive field 17
        self.out_layer = nn.Linear(self.hidden_dim * buffer_len, out_dim)

    def _get_linear(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, int(dim_in/10)),
                             nn.ReLU(),
                             nn.Linear(int(dim_in/10), dim_out))

    def _init_buffer(self, inputs):
        batch_size = inputs.shape[0]
        self.buffer = Variable(inputs.data.new(
            batch_size, self.buffer_len, self.buffer_dim).zero_())
        # TODO: init buffer with speaker indentity once we have it in action.

    def _update_buffer(self, buffer, context, memory):
        # VL paper uses memory/30 for no reason
        z = torch.cat([context, memory], 1)
        z = z.unsqueeze(1)
        z = torch.cat([z, self.buffer[:, :-1, :]], 1)
        u = self.buffer_proj_linear(z.view(z.shape[0], -1))
        self.buffer = torch.cat([u.unsqueeze(1), self.buffer[:, :-1, :]], 1)

    def forward(self, inputs, targets):
        greedy = not self.training
        assert (targets.size(1) * targets.size(2)) % self.out_dim == 0
        if targets.size(1) * targets.size(2) != self.out_dim:
            divisor = int((targets.size(1) * targets.size(2) / self.out_dim))
            targets = targets.view(targets.size(0), divisor, -1)
        # T x B x D
        targets = targets.transpose(0, 1)
        T_decoder = targets.shape[0]
        outputs, attns = [], []
        memory = Variable(inputs.data.new(
            inputs.shape[0], targets.shape[2]).zero_()).contiguous()
        self._init_buffer(inputs)
        t = 0
        while True:
            if t > 0:
                if greedy:
                    memory = outputs[-1].contiguous()
                else:
                    memory = targets[t-1].contiguous()
            # print(memory.shape)
            memory = self.memory_proj_linear(memory.view(memory.shape[0], -1))
            context, attn = self.attention(
                inputs, self.buffer.view(inputs.shape[0], -1))
            self._update_buffer(self.buffer, context,
                                memory.view(inputs.shape[0], -1))
            output = self.conv_layer(self.buffer)
            output = self.out_layer(output.view(output.shape[0], -1))
            outputs += [output]
            attns += [attn]
            t += 1
            if (not greedy and self.training) or (greedy and memory is not None):
                if t >= T_decoder:
                    break
            else:
                if t > 1 and is_end_of_frames(output, self.eps):
                    break
                elif t > self.max_decoder_steps:
                    print(" !! Decoder stopped with 'max_decoder_steps'. \
                          Something is probably wrong.")
                    break
        assert greedy or len(outputs) == T_decoder
        # Back to batch first
        attns = torch.stack(attns).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        return outputs, attns
