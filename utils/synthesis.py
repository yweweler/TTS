import io
import time
import librosa
import torch
import numpy as np
from TTS.utils.text import text_to_sequence
from TTS.utils.visual import visualize
from matplotlib import pylab as plt

hop_length = 250


def create_speech(m, s, CONFIG, use_cuda, ap):
    text_cleaner = [CONFIG.text_cleaner]
    seq = np.array(text_to_sequence(s, text_cleaner))
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_out, alignments, stop_tokens = m.forward(chars_var.long())
    mel_out = mel_out[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    return alignment, mel_out, stop_tokens


def tts(model, text, CONFIG, use_cuda, ap, figures=True):
    t_1 = time.time()
    alignment, spectrogram, stop_tokens = create_speech(model, text, CONFIG, use_cuda, ap) 
    print(" >  Run-time: {}".format(time.time() - t_1))
    if figures:                                                                                                         
        visualize(alignment, spectrogram, stop_tokens, text, ap.hop_length, CONFIG)                                                                       
    return alignment, spectrogram, stop_tokens