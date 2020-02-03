from pathlib import Path

import torch
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.LongTensor(range(0, max_len)).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    root_dir = Path(filename).parent
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        filepaths_and_text = [[str(root_dir / x[0]), x[1]] for x in filepaths_and_text]

    return filepaths_and_text


def to_device(inp, device):
    if hasattr(inp, 'to'):
        inp = inp.to(device)
    else:
        for i in range(len(inp)):
            inp[i] = to_device(inp[i], device)

    return inp
