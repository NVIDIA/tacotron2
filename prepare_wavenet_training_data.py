from scipy.io.wavfile import write
import librosa
import numpy as np
import argparse
import os
import sys
from hparams import create_hparams
from utils import load_wav_to_torch
from layers import TacotronSTFT
import torch

sr = 22050
max_wav_value=32768.0
trim_fft_size = 1024
trim_hop_size = 256
trim_top_db = 23

def get_audio(audio_path):
    data, sampling_rate = librosa.core.load(audio_path, sr)
    data = data / np.abs(data).max() * 0.999
    data_ = librosa.effects.trim(data, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
    #print(data_.max(), data_.min())
    return torch.FloatTensor(data_.astype(np.float32))

def get_mel(stft, audio):
    audio_norm = audio.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    #print(audio_norm.max(), audio_norm.min())
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.squeeze(0).transpose(0,1)
    return melspec

def _sign(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.sign(x)


def _log1p(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.log1p(x)


def _abs(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return np.abs(x)


def _asint(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return x.astype(np.int)


def _asfloat(x):
	#wrapper to support tensorflow tensors/numpy arrays
	isnumpy = isinstance(x, np.ndarray)
	isscalar = np.isscalar(x)
	return x.astype(np.float32)

def mulaw(x, mu=256):
	mu = 255
	return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)

def mulaw_quantize(x, mu=256):
	mu = 255
	y = mulaw(x, mu)
	return _asint((y + 1) / 2 * mu)

def save(out_dir, sentences, mels, audios):
    """
    training_data/audio/audio-1.npy|training_data/mels/mel-1.npy||<no_g>|여기에서 가까운 곳에 서점이 있나요?
    """
    with open(os.path.join(out_dir, 'map.txt'), 'w', encoding='utf-8') as file:
        for i in range(len(sentences)):
            audio_path = os.path.join(out_dir,'audio','audio-{}.npy'.format(i))
            audio = audios[i].astype(dtype=np.int16)
            mel_path = os.path.join(out_dir, 'mels', 'mel-{}.npy'.format(i))
            mel = mels[i]
            sentence = sentences[i]
            np.save(audio_path, audio)
            np.save(mel_path, mel)
            str = "{}|{}||<no_g>|{}\n".format(audio_path, mel_path, sentence)
            file.write(str)
        pass

def prepare_wavenet_training_data(hparams, out_dir, dataset):
    mel_dir = os.path.join(out_dir, 'mels')
    wav_dir = os.path.join(out_dir, 'audio')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    metadatas = open(os.path.join(dataset,'metadata.csv'),'r',encoding='utf-8').readlines()
    audio_paths = []
    sentences = []
    mels = []
    mus = []

    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

    for i, m in enumerate(metadatas):
        audio_path, sentence = m.strip().split('|')
        audio_path = os.path.join(dataset, 'wavs', audio_path)
        sentences.append(sentence)
        audio_paths.append(audio_path)

        audio = get_audio(audio_path)
        # print(audio.shape, audio.max(), audio.min())
        mel = get_mel(stft, audio)
        mels.append(mel)
        #print(mel.shape, mel.max(), mel.min())

        audio = audio.data.cpu().numpy()
        diff = len(audio) - hparams.hop_length * mel.size(0)
        if (diff >= 0):

            audio = audio[:-diff]
        else:
            audio = np.append(audio, [0.]*-diff)

        #print(len(audio)%hparams.hop_length ==0, len(audio)//mel.size(0) == hparams.hop_length, len(audio), len(audio)//mel.size(0))

        mu = mulaw_quantize(audio)
        mus.append(mu)
        # print(mu.shape, mu.max(), mu.min())
        if (i%100 == 0):
            print(i)

    save(out_dir, sentences, mels, mus)

    pass

if __name__ == "__main__":
    """
    usage
    python prepare_wavenet_training_data.py --dataset=nam-h --out_dir=training_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        help='file list to preprocess')
    parser.add_argument('-o', '--out_dir', type=str,
                        default='training_data', help='file list to preprocess')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()
    dataset = args.dataset
    out_dir = args.out_dir
    hparams = create_hparams(args.hparams)
    prepare_wavenet_training_data(hparams, out_dir, dataset)
