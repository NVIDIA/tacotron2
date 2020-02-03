import torch

import numpy as np

from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2

hparams = create_hparams()


if __name__ == '__main__':
    bs = 4
    tl = 24
    text_inputs = torch.LongTensor(np.random.randint(10, 100, size=(bs, tl)))
    text_lengths = torch.LongTensor([tl]*bs)
    mels = torch.FloatTensor(np.random.randn(bs, 80, 24))
    output_lengths = text_lengths
    max_len = 300

    inputs = (text_inputs, text_lengths, mels, max_len, output_lengths)

    tacotron = Tacotron2(hparams=hparams, device='cuda')

    print(tacotron(inputs))
