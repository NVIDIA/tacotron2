import torch
from torch.autograd import Variable
import numpy as np
from utils import load_wav_to_torch
from hparams import create_hparams
from layers import TacotronSTFT

def alignment_metric(alignments, input_lengths, output_lengths):
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y

    batch_size = alignments.size(0)
    optimums = torch.sqrt(input_lengths.double()**2 + output_lengths.double()**2)

    diagonalitys = torch.zeros(batch_size)
    val_sum = torch.zeros(1)
    for i in range(batch_size):
        dist = torch.zeros(1)
        for j in range(output_lengths[i]):
            value, cur_idx = torch.max(alignments[i][:][j], 0)
            val_sum += value
            if j==0:
                prev_idx = cur_idx
                continue
            else:
                dist += (1 + (cur_idx - prev_idx).pow(2)).float().pow(0.5)
                prev_idx = cur_idx
        diagonalitys[i] = Variable(dist/optimums[i])
    avg_prob = Variable(val_sum / torch.sum(output_lengths).float())
    diagonality = torch.mean(diagonalitys)
    return diagonality, avg_prob

def evaluation_metrics(stft, source_mels, target_mels):
    batch_size = source_mels.size(0)
    MCDs = torch.zeros(batch_size)
    f0s = None
    for i in range(batch_size):
        src_mel = source_mels[i].unsqueeze(0)
        src_mel = torch.clamp(src_mel, min=-4.0, max=4.0)
        dst_mel = target_mels[i].unsqueeze(0)
        dst_mel = torch.clamp(dst_mel, min=-4.0, max=4.0)
        MCDs[i] = MCD_from_mels(stft, src_mel, dst_mel)
        f0 = sqDiffF0_from_mels(stft, src_mel, dst_mel)
        f0s = f0 if f0s is None else torch.cat((f0s, f0), 0)

    avg_MCD = torch.mean(MCDs)
    avg_f0 = torch.mean(f0s)

    return avg_MCD, avg_f0

def melCepDist(srcMCC, dstMCC):
    # https://dsp.stackexchange.com/questions/56391/mel-cepstral-distortion
    diff = dstMCC - srcMCC
    return torch.sum((torch.sqrt( 2 * (diff**2) ) ))* (10.0/np.log(10)) * 1/diff.size(1)

def f0(MCC):
    #print(MCC.shape, MCC.max(), MCC.min())
    _, f0 = MCC.max(0)
    return f0

def MCD_from_mels(stft, srcMel, dstMel):
    srcMCC = stft.cepstrum_from_mel(srcMel)[0,:25,:]
    #print('srcMCC: ', srcMCC.max(), srcMCC.min())
    dstMCC = stft.cepstrum_from_mel(dstMel)[0,:25,:]
    #print('dstMCC: ', dstMCC.max(), dstMCC.min())
    MCD = melCepDist(srcMCC,dstMCC)
    log_MCD = torch.log10(torch.clamp(MCD,min=1e-5))
    return log_MCD

def sqDiffF0_from_mels(stft, srcMel, dstMel):
    srcMCC = stft.cepstrum_from_mel(srcMel).squeeze(0)
    dstMCC = stft.cepstrum_from_mel(dstMel).squeeze(0)
    srcF0 = f0(srcMCC)
    dstF0 = f0(dstMCC)
    diff = (dstF0 - srcF0).double()
    return torch.sqrt(diff**2)

def test_MCD_and_f0():
    hparams = create_hparams()
    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
    audio_path = 'kakao/1/1_0001.wav'
    mel_path = 'kakao/1/1_0001.mel.npy'
    srcMel = torch.from_numpy(np.load(mel_path)).unsqueeze(0)
    srcMel = torch.clamp(srcMel, -4.0, 4.0)
    # print(srcMel.shape,  srcMel.max(), srcMel.min())
    audio, sr = load_wav_to_torch(audio_path)
    # print(audio.shape, audio.max(), audio.min())
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    # print(audio_norm.shape, audio_norm.max(), audio_norm.min())
    dstMel = stft.mel_spectrogram(audio_norm)
    # print(dstMel.shape, dstMel.max(), dstMel.min())
    # mcc = stft.cepstrum_from_audio(audio_norm)
    # print('mcc', mcc.shape, mcc.max(), mcc.min())

    log_MCD = MCD_from_mels(stft, srcMel, dstMel)
    print(log_MCD.data, 'log')

    sqrtDiffF0 = sqDiffF0_from_mels(stft, srcMel, dstMel)
    print(sqrtDiffF0)
    meanSqrtDiffF0 = torch.mean(sqrtDiffF0)
    print(meanSqrtDiffF0.data, '100hz')

#alignment_metric()
if __name__ == "__main__":
    test_MCD_and_f0()


    #np.save('mel.npy' ,mel)