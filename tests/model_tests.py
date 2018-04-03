import unittest
import torch as T
import numpy as np

from TTS.models.tacotron import Tacotron


class VoiceLoopConvTests(unittest.TestCase):
    def test_in_out(self):
        layer = Tacotron(embedding_dim=256, linear_dim=1025, mel_dim=80, r=5)
        dummy = np.random.randint(4, 100, size=[4, 100])
        dummy_input = T.autograd.Variable(T.LongTensor(dummy))
        dummy_target = T.autograd.Variable(T.rand(4, 120, 80))

        mel, linear, align = layer(dummy_input, dummy_target)

        assert mel.shape[0] == 4
        assert mel.shape[1] == 120
        assert mel.shape[2] == 80
        assert linear.shape[0] == 4
        assert linear.shape[1] == 120
        assert linear.shape[2] == 1025
        assert align.shape[0] == 4, "size is not {} but {}".format(
            4, align.shape[0])
        assert align.shape[1] == 24,  "size is not {} but {}".format(
            24, align.shape[1])
        assert align.shape[2] == 100,  "size is not {} but {}".format(
            120, align.shape[2])
