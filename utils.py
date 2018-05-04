import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len).long().cuda()
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def load_filepaths_and_text(filename, sort_by_length, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]

    if sort_by_length:
        filepaths_and_text.sort(key=lambda x: len(x[1]))

    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous().cuda(async=True)
    return torch.autograd.Variable(x)
