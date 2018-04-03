import unittest
import torch as T
import numpy as np

from TTS.layers.tacotron import Prenet, CBHG, Decoder, Encoder
from TTS.layers.attention import AttentionLayer
from TTS.layers.conv import EncoderConv, DecoderConvWithBuffer
from layers.losses import L1LossMasked, _sequence_mask


class PrenetTests(unittest.TestCase):
    def test_in_out(self):
        layer = Prenet(128, out_features=[256, 128])
        dummy_input = T.autograd.Variable(T.rand(4, 128))

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 128


class CBHGTests(unittest.TestCase):
    def test_in_out(self):
        layer = CBHG(128, K=6, projections=[128, 128], num_highways=2)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 128))

        print(layer)
        output = layer(dummy_input)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256


class DecoderTests(unittest.TestCase):
    def test_in_out(self):
        layer = Decoder(in_features=256, memory_dim=80, r=2)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 256))
        dummy_memory = T.autograd.Variable(T.rand(4, 2, 80))

        output, alignment = layer(dummy_input, dummy_memory)

        assert output.shape[0] == 4
        assert output.shape[1] == 1, "size not {}".format(output.shape[1])
        assert output.shape[2] == 80 * 2, "size not {}".format(output.shape[2])


class EncoderTests(unittest.TestCase):
    def test_in_out(self):
        layer = Encoder(128)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 128))

        print(layer)
        output = layer(dummy_input)
        print(output.shape)
        assert output.shape[0] == 4
        assert output.shape[1] == 8
        assert output.shape[2] == 256  # 128 * 2 BiRNN


class L1LossMaskedTests(unittest.TestCase):
    def test_in_out(self):
        layer = L1LossMasked()
        dummy_input = T.autograd.Variable(T.ones(4, 8, 128).float())
        dummy_target = T.autograd.Variable(T.ones(4, 8, 128).float())
        dummy_length = T.autograd.Variable((T.ones(4) * 8).long())
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.shape[0] == 1
        assert len(output.shape) == 1
        assert output.data[0] == 0.0

        dummy_input = T.autograd.Variable(T.ones(4, 8, 128).float())
        dummy_target = T.autograd.Variable(T.zeros(4, 8, 128).float())
        dummy_length = T.autograd.Variable((T.ones(4) * 8).long())
        output = layer(dummy_input, dummy_target, dummy_length)
        assert output.data[0] == 1.0, "1.0 vs {}".format(output.data[0])

        dummy_input = T.autograd.Variable(T.ones(4, 8, 128).float())
        dummy_target = T.autograd.Variable(T.zeros(4, 8, 128).float())
        dummy_length = T.autograd.Variable((T.arange(5, 9)).long())
        mask = ((_sequence_mask(dummy_length).float() - 1.0)
                * 100.0).unsqueeze(2)
        output = layer(dummy_input + mask, dummy_target, dummy_length)
        assert output.data[0] == 1.0, "1.0 vs {}".format(output.data[0])


class AttentionLayerTests(unittest.TestCase):
    def test_in_out(self):
        layer = AttentionLayer(32, 64, 15*96)
        dummy_annot = T.autograd.Variable(T.rand(4, 8, 64))
        dummy_query = T.autograd.Variable(T.rand(4, 15, 96))

        output, alignment = layer(dummy_annot, dummy_query.view(4, 1, 15*96))

        assert output.shape[0] == 4
        assert output.shape[1] == 64, "size not {}".format(output.shape[1])


class EncoderConvTests(unittest.TestCase):
    def test_in_out(self):
        layer = EncoderConv(vocab_size=100, embed_dim=128,
                            out_dim=86, hidden_dim=64)
        dummy = np.random.randint(4, 100, size=[4, 12])
        dummy_input = T.autograd.Variable(T.LongTensor(dummy))

        output = layer.forward(dummy_input)

        assert output.shape[0] == 4, "size not {}".format(output.shape[0])
        assert output.shape[1] == 12, "size not {}".format(output.shape[1])
        assert output.shape[2] == 86, "size not {}".format(output.shape[2])


class DecoderConvWithBufferTests(unittest.TestCase):
    def test_in_out(self):
        layer = DecoderConvWithBuffer(
            in_dim=60, out_dim=32*5, hidden_dim=24, buffer_len=17, memory_dim=16)
        dummy_input = T.autograd.Variable(T.rand(4, 8, 60).float())
        dummy_target = T.autograd.Variable(T.ones(4, 25, 32).float())

        layer = layer.train()
        output, attns = layer.forward(dummy_input, dummy_target)

        assert output.shape[0] == 4, "size not {}".format(output.shape[0])
        assert output.shape[1] == 5, "size not {}".format(output.shape[1])
        assert output.shape[2] == 32*5, "size not {}".format(output.shape[2])

        layer = layer.eval()
        output, attns = layer.forward(dummy_input, dummy_target)

        assert output.shape[0] == 4, "size not {}".format(output.shape[0])
        assert output.shape[1] == 5, "size not {}".format(output.shape[1])
        assert output.shape[2] == 32*5, "size not {}".format(output.shape[2])
