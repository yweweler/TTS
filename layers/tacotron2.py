# coding: utf-8
# TODO: Zoneout LSTMs and Dropout Convs
import torch
from torch import nn
from .attention import AttentionCell

# from .custom_layers import ZoneOutCell


class Prenet(nn.Module):
    r""" Prenet as explained at https://arxiv.org/abs/1703.10135.
    It creates as many layers as given by 'out_features'

    Args:
        in_features (int): size of the input vector
        out_features (int or list): size of each output sample.
            If it is a list, for each value, there is created a new layer.
    """

    def __init__(self, in_features, out_features=[256, 128], dropout=0.0):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.relu = nn.ReLU()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, inputs):
        for linear in self.layers:
            if self.dropout.p > 0:
                inputs = self.dropout(self.relu(linear(inputs)))
            else:
                inputs = self.relu(linear(inputs))
        return inputs


class BatchNormConv1d(nn.Module):
    r"""A wrapper for Conv1d with BatchNorm. It sets the activation
    function between Conv and BatchNorm layers. BatchNorm layer
    is initialized with the TF default values for momentum and eps.

    Args:
        in_channels: size of each input sample
        out_channels: size of each output samples
        kernel_size: kernel size of conv filters
        stride: stride of conv filters
        padding: padding of conv filters

    Shapes:
        - input: batch x dims
        - output: batch x dims
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class EncoderConvStack(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_layers=3,
                 kernel_size=5):
        super(EncoderConvStack, self).__init__()
        self.layers = []
        self.padding = (kernel_size - 1) // 2
        self.stride = 1
        for idx in range(num_layers):
            if idx == 0:
                layer = BatchNormConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=nn.ReLU)
            else:
                layer = BatchNormConv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=nn.ReLU)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        z = x
        z = z.transpose(2, 1)
        for layer in self.layers:
            z = layer(z)
        z = z.transpose(2, 1)
        return z


class Encoder(nn.Module):
    r"""Encapsulate Prenet and CBHG modules for encoder"""

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.conv_stack = EncoderConvStack(
            in_channels=in_features, out_channels=in_features)
        self.lstm = nn.LSTM(
            in_features,
            in_features // 2,
            1,
            batch_first=True,
            bidirectional=True)

    def forward(self, inputs):
        r"""
        Args:
            inputs (FloatTensor): embedding features

        Shapes:
            - inputs: batch x time x in_features
            - outputs: batch x time x in_features
        """
        z = self.conv_stack(inputs)
        self.lstm.flatten_parameters()
        out = self.lstm(z)
        return out[0]


class PostConvStack(nn.Module):
    def __init__(self,
                 in_channels=80,
                 hidden_channels=512,
                 out_channels=80,
                 num_layers=5,
                 kernel_size=5):
        super(PostConvStack, self).__init__()
        self.layers = []
        self.padding = (kernel_size - 1) // 2
        self.stride = 1
        for idx in range(num_layers):
            if idx == 0:
                layer = BatchNormConv1d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=nn.Tanh)
            elif idx == num_layers - 1:
                layer = BatchNormConv1d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=None)
            else:
                layer = BatchNormConv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    activation=nn.Tanh)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        z = x
        z = z.transpose(2, 1)
        for layer in self.layers:
            z = layer(z)
        z = z.transpose(2, 1)
        return z


class Decoder(nn.Module):
    r"""Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
    """

    def __init__(self, in_features, memory_dim, r, lstm_size=512, prenet_size=256, atten_size=128):
        super(Decoder, self).__init__()
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.memory_dim = memory_dim
        self.lstm_size = lstm_size
        self.prenet_size = prenet_size
        self.atten_size = atten_size
        # memory -> |Prenet| -> processed_memory
        self.prenet = Prenet(
            memory_dim * r, out_features=[prenet_size, prenet_size], dropout=0.5)
        # (context(t-1), processed_memory(t)) -> |Attention| -> attention, context(t), rnn_hidden
        self.attention_rnn = AttentionCell(
            input_dim=lstm_size,
            atten_dim=atten_size,
            annot_dim=in_features,
            align_model='ls')

        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        # self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        # (context(t), processed_memory) -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList([
            nn.LSTMCell(in_features + prenet_size, lstm_size),
            nn.LSTMCell(lstm_size, lstm_size)
        ])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(lstm_size + in_features, memory_dim * r)
        self.stopnet = nn.Sequential(
            nn.Linear(lstm_size + in_features, 1, bias=False), nn.Sigmoid())

    def forward(self, inputs, memory=None, mask=None):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.

        Args:
            inputs: Encoder outputs.
            memory (None): Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask (None): Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        B = inputs.size(0)
        T = inputs.size(1)
        # Run greedy decoding if memory is None
        greedy = not self.training
        if memory is not None:
            # Grouping multiple frames if necessary
            if memory.size(-1) == self.memory_dim:
                memory = memory.view(B, memory.size(1) // self.r, -1)
                " !! Dimension mismatch {} vs {} * {}".format(
                    memory.size(-1), self.memory_dim, self.r)
            T_decoder = memory.size(1)
        # go frame as zeros matrix
        initial_memory = inputs.data.new(B, self.memory_dim * self.r).zero_()
        # decoder states
        decoder_rnns_states = [[
            inputs.data.new(B, self.lstm_size).zero_(),
            inputs.data.new(B, self.lstm_size).zero_()
        ], [
            inputs.data.new(B, self.lstm_size).zero_(),
            inputs.data.new(B, self.lstm_size).zero_()
        ]]
        current_context_vec = inputs.data.new(B, self.in_features).zero_()
        # Time first (T_decoder, B, memory_dim)
        if memory is not None:
            memory = memory.transpose(0, 1)
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        memory_input = initial_memory
        while True:
            if t > 0:
                if memory is None:
                    memory_input = outputs[-1]
                else:
                    memory_input = memory[t - 1]
            # Prenet
            processed_memory = self.prenet(memory_input)
            # Decoder RNNs
            decoder_rnns_input = torch.cat(
                (processed_memory, current_context_vec), -1)
            decoder_rnns_output = decoder_rnns_input
            for idx, layer in enumerate(self.decoder_rnns):
                decoder_rnns_output, decoder_rnns_cell = layer(
                    decoder_rnns_output, decoder_rnns_states[idx])
                decoder_rnns_states[idx][0] = decoder_rnns_output
                decoder_rnns_states[idx][1] = decoder_rnns_cell
            # Attention
            attention_cat = torch.cat(
                (attention.unsqueeze(1), attention_cum.unsqueeze(1)), dim=1)
            current_context_vec, attention = self.attention_rnn(
                decoder_rnns_output, inputs, attention_cat, mask)
            attention_cum += attention
            # predict mel vectors from decoder vectors
            decoder_proj_input = torch.cat(
                (decoder_rnns_output, current_context_vec), dim=1)
            output = self.proj_to_mel(decoder_proj_input)
            # predict stop token
            stop_token = self.stopnet(decoder_proj_input)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            if memory is not None:
                if t >= T_decoder:
                    break
            else:
                if t > inputs.shape[1] / 4 and stop_token > 0.6:
                    break
                elif t > self.max_decoder_steps:
                    print("   | > Decoder stopped with 'max_decoder_steps")
                    break
        # assert greedy or len(outputs) == T_decoder
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        return outputs, attentions, stop_tokens