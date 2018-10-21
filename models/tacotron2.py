# coding: utf-8
import torch
from torch import nn
from utils.text.symbols import symbols
from layers.tacotron2 import Prenet, Encoder, Decoder, PostConvStack


class Tacotron2(nn.Module):
    def __init__(self,
                 mel_dim=80,
                 r=1,
                 padding_idx=None,
                 embedding_size=512,
                 post_conv_stack_size=512,
                 decoder_lstm_size=1024,
                 decoder_prenet_size=256,
                 atten_size=128,
                 ):
        super(Tacotron2, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.embedding = nn.Embedding(
            len(symbols), embedding_size, padding_idx=padding_idx)
        print(" | > Number of characters : {}".format(len(symbols)))
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(embedding_size, mel_dim, r, lstm_size=decoder_lstm_size, prenet_size=decoder_prenet_size, atten_size=atten_size)
        self.postnet = PostConvStack(in_channels=mel_dim*r, hidden_channels=post_conv_stack_size, out_channels=mel_dim*r)
    

    def forward(self, characters, mel_specs=None, mask=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        # batch x time x dim
        encoder_outputs = self.encoder(inputs)
        # batch x time x dim*r
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        mel_output_res = self.postnet(mel_outputs)     
        mel_outputs += mel_output_res
        return mel_outputs, alignments, stop_tokens
