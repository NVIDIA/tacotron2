import librosa
import argparse

sr = 22050
max_wav_value=32768.0
trim_fft_size = 1024
trim_hop_size = 256
trim_top_db = 23

def preprocess_audio(file_name):
    f = open(file_name,'r',encoding='utf-8')
    R = f.readlines()
    f.close()

    L = []
    for i, r in enumerate(R):
        wav_file = r.split('|')[0]
        data, sampling_rate = librosa.core.load(wav_file, sr)
        if(len(data) >= 14848):
            L.append(r)

    skiped_file_name = file_name.split('.')[0]+'_skiped.txt'
    f = open(skiped_file_name,'w',encoding='utf-8')
    f.writelines(L)
    f.close()

if __name__ == "__main__":
    """
    usage
    python preprocess_audio.py -f=filelists/nam-h_test_filelist.txt,filelists/nam-h_train_filelist.txt,filelists/nam-h_val_filelist.txt -s=3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str,
                        help='map.txt file name')
    args = parser.parse_args()

    preprocess_audio(args.file_name)
