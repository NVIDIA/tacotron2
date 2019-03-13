import numpy as np
import argparse
import os

def merge(source_metadata, target_metadata, out_dir=''):
    concatinated_metas = []
    with open(source_metadata, 'r', encoding='utf-8') as file:
        source_metas = file.readline()
    with open(target_metadata, 'r', encoding='utf-8') as file:
        target_metas = file.readline()

    with open(os.path.join(out_dir,'vc_metadata.csv'),'w',encoding='utf-8') as file:
        for i, source_meta in enumerate(source_metas):
            source_meta = source_meta.strip()
            target_meta = target_metas[i].strip()
            concatinated_metas.append("{}|{}\n".format(source_meta,target_meta))
        file.writelines(concatinated_metas)
    pass

if __name__ == "__main__":
    """
    usage
    python merge_m2m_metadatas.py --out_dir=. --source_metadata=park_inferece/metadata.csv --target_metadata=park_m2m/metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='',type=str, help='output path for voice conversion metadata')
    parser.add_argument('-s', '--source_metadata', type=str, help='source melspectrogram dataset meta')
    parser.add_argument('-t', '--target_metadata', type=str, help='target melspectrogram dataset meta')
    args = parser.parse_args()
    merge(args.source_metadata, args.target_metadata, args.out_dir)