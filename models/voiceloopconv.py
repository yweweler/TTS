# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
from TTS.utils.text.symbols import symbols
from TTS.layers.conv import EncoderConv, DecoderConvWithBuffer


class VoiceLoopConv(nn.Module):
    def __init__(self, embedding_dim=256, linear_dim=1025, mel_dim=80, r=5):
                 
        super(VoiceLoopConv, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        # Trying smaller std
        self.encoder = EncoderConv(len(symbols), embedding_dim, embedding_dim, embedding_dim)
        self.decoder = DecoderConvWithBuffer(in_dim=embedding_dim, out_dim=mel_dim*r, hidden_dim=embedding_dim, buffer_len=17, memory_dim=mel_dim)
        self.last_linear = nn.Sequential(nn.Linear(mel_dim, 256), nn.ReLU(), nn.Linear(256, linear_dim))

    def forward(self, characters, mel_specs=None):
        B = characters.size(0)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(characters)
        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(encoder_outputs, mel_specs)
        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.last_linear(mel_outputs)
        return mel_outputs, linear_outputs, alignments
