import torch
from torch.autograd import Variable
import numpy as np

import wave
from scipy.io.wavfile import read
from layers import cepstral
from parabolic import parabolic
from scipy.signal import blackmanharris



def alignment_metric(alignments):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_len = torch.from_numpy(np.array(alignments[0].shape[1])).float()
    y_len = torch.from_numpy(np.array(alignments[0].shape[0])).float()

    # Compute the squared distances
    optimum = np.array((x_len.pow(2) + y_len.pow(2)).pow(0.5))
    dist = torch.zeros(1)
    val_sum = torch.zeros(1)
    for i in range(np.int(y_len)):
        value, cur_idx = torch.max(alignments[0][i], 0)
        val_sum += value
        if i==0:
            prev_idx = cur_idx
            continue
        else:
            dist += (1 + (cur_idx - prev_idx).pow(2)).float().pow(0.5)
            prev_idx = cur_idx

    avg_prob = Variable(val_sum /y_len).float()
    optimum = torch.from_numpy(optimum)
    rate = Variable(dist/optimum)

    return rate, avg_prob


def MCD(source_sound, syn_sound):
    sourc_cep = source_sound.cepstral()
    syn_cep = syn_sound.cepstral()

    mcd = 10 * ((2*torch.sum(sourc_cep-syn_cep).pow(2)).pow(0.5))/torch.log(10)

    return mcd


def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = np.fft.rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return torch.from_numpy(np.array(fs * true_i / len(windowed))).float()


def f0(wav):
    nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()

    # Inititalize a fundamental frequency
    freqs = torch.tensor([])
    up = framerate // 80
    down = framerate // 270
    d = framerate / 270.0

    # Number of frames per window
    window_size = 1024

    # Create a window function
    window = np.hamming(window_size)

    # Iterate over the wave file frames
    for i in range(nframes // window_size):
        # Reading n=window_size frames from the wave file
        content = wav.readframes(window_size)

        # Converting array of bytes to array of integers according to sampwidth. If stereo only the first channel is picked
        samples = np.fromstring(content, dtype=types[sampwidth])[0::nchannels]

        # Applying window function for our samples
        samples = torch.from_numpy(window * samples)

        # Calculating spectrum of a signal frame as fft with n=window_size
        #spectrum = np.fft.fft(samples, n=window_size)

        # Calculating cepstrum as ifft(log(abs(spectrum))))
        #cepstrum = np.fft.ifft(np.log(np.abs(spectrum))).real

        cepstrum = cepstral(samples)

        _, idx = torch.max(cepstrum[down:up])

        # Calculating fundamental frequency by finding peak
        fund_freq = torch.from_numpy(np.array(framerate)).float() * cepstrum.shape[0] / (idx + d) / cepstrum.shape[0]
        freqs = torch.cat(freqs, fund_freq)

    return torch.from_numpy(np.array(freqs))


def cal_fft(src_sound, syn_sound):
    src_f0 = f0(src_sound)
    syn_f0 = f0(syn_sound)
    return Variable(torch.sum(((src_f0 - syn_f0).pow(2))/src_f0.shape[0]).pow(0.5))

#src_sound = wave.open("C:/Users/chme/Desktop/Voice_AI/wavenet-audio-mel_wiener_.wav", mode='r')
#syn_sound = wave.open("C:/Users/chme/Desktop/Voice_AI/wavenet-audio-mel_.wav", mode='r')
#print(cal_fft(src_sound, syn_sound)) #, MCD(source_sound, syn_sound))
