from scipy.io.wavfile import write
import librosa
import numpy as np

sr = 22050
max_wav_value=32768.0
file_list = ['filelists/nam-h_test_filelist.txt', 'filelists/nam-h_train_filelist.txt', 'filelists/nam-h_val_filelist.txt']
#file_list = ['filelists/ljs_audio_text_test_filelist.txt', 'filelists/ljs_audio_text_train_filelist.txt', 'filelists/ljs_audio_text_val_filelist.txt']
# M-AILABS (and other datasets) trim params
trim_fft_size = 1024
trim_hop_size = 256
trim_top_db = 23
# wav_file = '0000000.wav'
silence_audio_size = trim_hop_size*3

def run():
    for F in file_list:
        f = open(F)
        R = f.readlines()
        f.close()
        print('='*5+F+'='*5)

        for i, r in enumerate(R):
            wav_file = r.split('|')[0]
            data, sampling_rate = librosa.core.load(wav_file, sr)
            data_= librosa.effects.trim(data, top_db= trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
            data_ = data_*max_wav_value          
            data_ = np.append(data_, [0.]*silence_audio_size)
            data_ = data_.astype(dtype=np.int16)
            write(wav_file, sr, data_)
            #print(len(data),len(data_))
            if(i%100 == 0):
                print (i)

if __name__ == "__main__":
    # for test
    # wav_file = '0000000.wav'
    # data, sampling_rate = librosa.core.load(wav_file, sr)
    # data_ = librosa.effects.trim(data, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
    # print(len(data), len(data_))
    # write('test1.wav', sampling_rate, data)
    # write('test2.wav', sampling_rate, data_)
    #
    # # for test 2
    # wav_file = '0000000.wav'
    # wav_file2 = 'test1.wav'
    # wav_file3 = 'test2.wav'
    #
    # sr1, data1= read(wav_file)
    # sr2, data2 = read(wav_file2)
    # sr3, data3 = read(wav_file3)
    #
    # print('sr1:{}, sr2:{}, sr3:{}'.format(sr1, sr2, sr3))
    # print('{} {} {}'.format(len(data1), len(data2), len(data3)))
    # print('{} {} {}'.format(data1.min(), data2.min(), data3.min()))
    # print('{} {} {}'.format(data1.max(), data2.max(), data3.max()))

    # for run
    run()
