from scipy.io.wavfile import write
import librosa

sr = 22050
max_wav_value=32768.0
file_list = ['filelists/nam-h_test_filelist.txt', 'filelists/nam-h_train_filelist.txt', 'filelists/nam-h_val_filelist.txt']

for F in file_list:
    f = open(F)
    R = f.readlines()
    f.close()
    print('='*5+F+'='*5)

    for i, r in enumerate(R):
        wav_file = r.split('|')[0]
        data, sampling_rate = librosa.core.load(wav_file, sr)
        write(wav_file, sr, data*max_wav_value)