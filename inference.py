import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
#import IPython.display as ipd

import numpy as np
import time
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig('tmp.png')

hparams = create_hparams("distributed_run=False,mask_padding=False")
hparams.sampling_rate = 22050
hparams.filter_length = 1024
hparams.hop_length = 256
hparams.win_length = 1024

# checkpoint_path = "/home/hwak1234/projects/tacotron2/outdir/checkpoint_15000"
# hparams.n_symbols = 149
checkpoint_path = "/home/hwak1234/projects/tacotron2/nam_h_ep2/checkpoint_61500"
hparams.n_symbols = 80
model = load_model(hparams)
try:
    model = model.module
except:
    pass
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path)['state_dict'].items()})
_ = model.eval()

# text = "This is an example of text to speech synthesis after 14 hours training."
# sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
text = "테스트 문장이야."
sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
print(sequence)
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

stime = time.time()
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.data.cpu().numpy()[0],
           mel_outputs_postnet.data.cpu().numpy()[0],
           alignments.data.cpu().numpy()[0].T))
inf_time =  time.time() - stime

stime = time.time()
taco_stft = TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    sampling_rate=hparams.sampling_rate)
mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
spec_from_mel_scaling = 1000
spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]),
                       taco_stft.stft_fn, 60)
waveform = waveform[0].data.cpu().numpy()
# waveform = (waveform - np.min(waveform))/(np.max(waveform) - np.min(waveform))*hparams.max_wav_value
# waveform = waveform.astype('int16')

dec_time = time.time() - stime
len_audio = float(len(waveform))/float(hparams.sampling_rate)
str = "audio length: {:.2f} sec,  inference time: {:.2f} sec,  decoding time: {:.2f}".format(len_audio, inf_time, dec_time)

print(len(waveform), hparams.sampling_rate)
print(str)

write("temp.wav", hparams.sampling_rate, waveform)