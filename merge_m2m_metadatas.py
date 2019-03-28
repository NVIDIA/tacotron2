import numpy as np
import argparse
import os

def merge(source_metadata, target_metadata, out_dir=''):
    concatinated_metas = []
    with open(source_metadata, 'r', encoding='utf-8') as file:
        source_metas = file.readlines()
    with open(target_metadata, 'r', encoding='utf-8') as file:
        target_metas = file.readlines()
    print(len(source_metas),len(target_metas))
    with open(os.path.join(out_dir,'vc_metadata.csv'),'w',encoding='utf-8') as file:
        for source_meta in source_metas:
            s_mel_path, s_text = source_meta.strip().split('|')
            for i, target_meta in enumerate(target_metas):
                t_mel_path, t_text = target_meta.strip().split('|')
                if(t_text in s_text and s_text in t_text):
                    concatinated_metas.append("{}|{}|{}|{}\n".format(s_mel_path,s_text,t_mel_path,t_text))
                    target_metas.pop(i)
                    break
        file.writelines(concatinated_metas)
    pass

if __name__ == "__main__":
    """
    usage
    python merge_m2m_metadatas.py --out_dir=. --source_metadata=park_inference/metadata.csv --target_metadata=park_m2m/metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='',type=str, help='output path for voice conversion metadata')
    parser.add_argument('-s', '--source_metadata', type=str, help='source melspectrogram dataset meta')
    parser.add_argument('-t', '--target_metadata', type=str, help='target melspectrogram dataset meta')
    args = parser.parse_args()
    merge(args.source_metadata, args.target_metadata, args.out_dir)