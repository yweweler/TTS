# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
from TTS.utils.text.symbols import symbols


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
        self.conv1 = nn.Conv1d(in_features, hidden_features, self.kernel_size, dilation=self.dilation, padding=self.dil_padding)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.conv2 = nn.Conv1d(hidden_features, out_features, self.kernel_size, dilation=self.dilation, padding=self.dil_padding)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.scale = nn.Conv1d(in_features, out_features, 1)
    
    def forward(self, x):
        x_ = self.relu(self.bn1(self.conv1(x)))
        x_ = self.bn2(self.conv2(x_))
        assert x.shape[2] == x_.shape[2], " >  ! Size mismatch {} vs {}".format(x.shape, x_.shape)
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
            if i == 0:
                self.blocks.append(Conv1dBlock(in_features, self.hidden_features, self.hidden_features, kernel_size=kernel_size, dilation=dil))
            elif i == len(dilations)-1:
                self.block2 = Conv1dBlock(in_features, out_features, self.hidden_features, kernel_size=kernel_size, dilation=dil)
            else:
                self.blocks.append(Conv1dBlock(self.hidden_features, self.hidden_features, self.hidden_features, kernel_size=kernel_size, dilation=dil))
        self.blocks = nn.ModuleList(self.blocks)
                
    def forward(self, x):
        # Needed to perform conv1d on time-axis
        # (B, in_features, T_in)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return x
        

class MemoryBuffer(nn.Module):
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim,
                                      max_norm=1.0)
        self.conv_bank = Conv1dBank(embed_dim, in_features, in_features,
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
    

    
    
class Decoder(nn.Module):
    def __init__(self):
        
    def forward(self, inputs)


