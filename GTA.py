import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from fp16_optimizer import FP16_Optimizer

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from preprocess_audio import silence_audio_size, trim_hop_size

silence_mel_size = silence_audio_size/trim_hop_size

def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed")
    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # Initialize distributed communication
    torch.distributed.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams) # trainset.__getitem__(index) = (text, mel), text in [num_char], mel in [num_mel, ceil((len(audio)+1)/hop_length)]
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step) #
    train_sampler = DistributedSampler(trainset) \
        if hparams.distributed_run else None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, trainset


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model = batchnorm_to_float(model.half())
        model.decoder.attention_layer.score_mask_value = float(finfo('float16').min)
    if hparams.distributed_run:
        model = DistributedDataParallel(model)
    elif torch.cuda.device_count() > 1:
        model = DataParallel(model)
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def GTA_Synthesis(output_directory, checkpoint_path, n_gpus,
          rank, group_name, hparams):

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    model = load_model(hparams)
    train_loader, valset, collate_fn, train_set = prepare_dataloaders(hparams)
    # Load checkpoint if one exists
    assert checkpoint_path is not None
    if checkpoint_path is not None:
        model = warm_start_model(checkpoint_path, model)
        model.eval()
    if hparams.distributed_run or torch.cuda.device_count() > 1:
        batch_parser = model.module.parse_batch
    else:
        batch_parser = model.parse_batch
        # ================ MAIN TRAINNIG LOOP! ===================
    f = open(os.path.join(output_directory, 'map.txt'),'w', encoding='utf-8')
    os.makedirs(os.path.join(output_directory,'mels'), exist_ok=True)

    total_number_of_data = len(train_set.audiopaths_and_text)
    max_itter = int(total_number_of_data/hparams.batch_size)
    remainder_size = total_number_of_data % hparams.batch_size
    
    for i, batch in enumerate(train_loader):
        batch_size = hparams.batch_size if i is not max_itter else remainder_size

        # get wavefile path
        audiopaths_and_text = train_set.audiopaths_and_text[i*hparams.batch_size:i*hparams.batch_size + batch_size]
        audiopaths = [ x[0] for x in audiopaths_and_text] # file name list


        # get len texts
        indx_list = np.arange(i*hparams.batch_size, i*hparams.batch_size + batch_size).tolist()
        len_text_list = []
        for batch_index in indx_list:
            text, _ = train_set.__getitem__(batch_index)
            len_text_list.append(text.size(0))
        _, input_lengths, _, _, output_lengths = batch # output_lengths: orgnal mel length
        input_lengths_, ids_sorted_decreasing = torch.sort(torch.LongTensor(len_text_list), dim=0, descending=True)
        ids_sorted_decreasing = ids_sorted_decreasing.numpy() # ids_sorted_decreasing, original index

        org_audiopaths = [] # orgnal_file_name
        mel_paths = []
        for k in range(batch_size):
            d = audiopaths[ids_sorted_decreasing[k]]
            org_audiopaths.append(d)
            mel_paths.append(d.replace('wav','mel'))

        x, _ = batch_parser(batch)
        _, mel_outputs_postnet, _, _ = model(x)
        mel_outputs_postnet = mel_outputs_postnet.data.cpu().numpy()

        for k in range(batch_size):
            wav_path = org_audiopaths[k]
            mel_path = mel_paths[k]+'.npy'
            map = "{}|{}\n".format(wav_path,mel_path)
            f.write(map)

            mel = mel_outputs_postnet[k,:,:output_lengths[k]-silence_mel_size]
            np.save(mel_path, mel)
        print('compute and save GTA melspectrograms in {}th batch'.format(i))
    f.close()

if __name__ == '__main__':
    # run example
    # python GTA.py -o=nam-h -c=nam_h_ep8/checkpoint_50000
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    GTA_Synthesis(args.output_directory, args.checkpoint_path, args.n_gpus, args.rank, args.group_name, hparams)
