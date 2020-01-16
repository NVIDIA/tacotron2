"""
This code was developed with reference to https://github.com/Rayhane-mamah/Tacotron-2.
"""
from scipy.io.wavfile import write
import librosa
import numpy as np
import argparse

sr = 22050
max_wav_value=32768.0
trim_fft_size = 1024
trim_hop_size = 256

# These are control parameters for trimming and skipping
trim_top_db = 23
skip_len = 14848

def preprocess_audio(file_list, silence_audio_size, pre_emphasis=False):
    for F in file_list:
        f = open(F, encoding='utf-8')
        R = f.readlines()
        f.close()
        print('='*5+F+'='*5)

        for i, r in enumerate(R):
            wav_file = r.split('|')[0]
            data, sampling_rate = librosa.core.load(wav_file, sr)
            data = data / np.abs(data).max() *0.999
            data_= librosa.effects.trim(data, top_db= trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
            if (pre_emphasis):
                data_ = np.append(data_[0], data_[1:] - 0.97 * data_[:-1])
                data_ = data_ / np.abs(data_).max() * 0.999
            data_ = data_ * max_wav_value
            data_ = np.append(data_, [0.]*silence_audio_size)
            data_ = data_.astype(dtype=np.int16)
            write(wav_file, sr, data_)
            #print(len(data),len(data_))
            if(i%100 == 0):
                print (i)

def remove_short_audios(file_name):
    f = open(file_name,'r',encoding='utf-8')
    R = f.readlines()
    f.close()

    L = []
    for i, r in enumerate(R):
        wav_file = r.split('|')[0]
        data, sampling_rate = librosa.core.load(wav_file, sr)
        if(len(data) >= skip_len):
            L.append(r)
        if (i % 100 == 0):
            print(i)
    tmp = file_name.split('.')
    tmp.insert(1,'_skipped.')
    skipped_file_name = "".join(tmp)
    f = open(skipped_file_name,'w',encoding='utf-8')
    f.writelines(L)
    f.close()

if __name__ == "__main__":
    """
    usage
    python preprocess_dataset.py -f=metadata.csv -s=5 -t -p -r
    python preprocess_dataset.py -f=metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_list', type=str,
                        help='Metadata file list to preprocess')
    parser.add_argument('-s', '--silence_padding', type=int, default=0,
                        help='Adding silence padding at the end of each audio, silence audio size is hop_length * silence padding')
    parser.add_argument('-p', '--pre_emphasis', action='store_true',
                        help="Doing pre_emphasis")
    parser.add_argument('-t', '--trimming', action='store_true',
                        help="Doing trimming audios")
    parser.add_argument('-r', '--remove_short_audios',action='store_true',
                        help="Removing short audios in metadata file")
    args = parser.parse_args()
    file_list = args.file_list.split(',')
    silence_audio_size = trim_hop_size * args.silence_padding


    preprocess_audio(file_list, silence_audio_size, args.pre_emphasis)

    if(args.remove_short_audios):
        for f in file_list:
            remove_short_audios(f)