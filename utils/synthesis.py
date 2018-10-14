import io
import time
import librosa
import torch
import numpy as np
from TTS.utils.text import text_to_sequence
from TTS.utils.visual import visualize
from matplotlib import pylab as plt


def create_speech(m, s, CONFIG, use_cuda, ap):
    text_cleaner = [CONFIG.text_cleaner]
    seq = np.array(text_to_sequence(s, text_cleaner))
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_spec, alignments, stop_tokens = m.forward(chars_var.long())
    mel_spec = mel_spec[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    wav = ap.inv_mel_spectrogram(mel_spec.T)
    wav = wav[:ap.find_endpoint(wav)]
    out = io.BytesIO()
    ap.save_wav(wav, out)
    return wav, alignment, mel_spec, stop_tokens
