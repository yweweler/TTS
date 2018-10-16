import os
import unittest
import numpy as np
import torch as T
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config

file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))

class TestAudio(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAudio, self).__init__(*args, **kwargs)
        self.ap = AudioProcessor(**c.audio)

    def test_normalize(self):
        x = (np.random.rand(10, 10) - 1) * 100
        x_old = x

        self.ap.signal_norm = True
        self.ap.symmetric_norm = False
        self.ap.max_norm = 4.0
        x_norm = self.ap._normalize(x)
        assert (x_old - x).sum() == 0
        assert x_norm.max() <= self.ap.max_norm, x_norm.max()
        assert x_norm.min() >= 0, x_norm.min()
        x_ = self.ap._denormalize(x_norm)
        assert (x - x_).sum() < 1e-3

        self.ap.signal_norm = True
        self.ap.symmetric_norm = False
        self.ap.max_norm = 1.0
        x_norm = self.ap._normalize(x)
        assert (x_old - x).sum() == 0
        assert x_norm.max() <= self.ap.max_norm, x_norm.max()
        assert x_norm.min() >= 0, x_norm.min()
        x_ = self.ap._denormalize(x_norm)
        assert (x - x_).sum() < 1e-3

        self.ap.signal_norm = True
        self.ap.symmetric_norm = True
        self.ap.max_norm = 1.0
        x_norm = self.ap._normalize(x)
        assert (x_old - x).sum() == 0
        assert x_norm.max() <= self.ap.max_norm, x_norm.max()
        assert x_norm.min() >= -self.ap.max_norm, x_norm.min()
        assert x_norm.min() < 0, x_norm.min()
        x_ = self.ap._denormalize(x_norm)
        assert (x - x_).sum() < 1e-3
