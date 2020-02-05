import inspect
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from tacotron2.factory import Factory
from tacotron2.hparams import HParams
from tacotron2.layers import TacotronSTFT
from tacotron2.utils import load_wav_to_torch, load_filepaths_and_text


class TextMelLoader(torch.utils.data.Dataset):
    """Texts and mel-spectrograms dataset
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, meta_file_path: Path, tokenizer_class_name: str, load_mel_from_disk: bool, max_wav_value,
                 sampling_rate,  filter_length, hop_length, win_length, n_mel_channels, mel_fmin, mel_fmax,
                 n_frames_per_step):
        """
        :param meta_file_path: Path, value separated text meta-file which has two fields:
            - relative (from the meta-file itself) path to the wav audio sample
            - audio sample text
            Fields must be separated by '|' symbol
        :param tokenizer_class_name: str, tokenizer class name. Must be importable from tacotron2.tokenizers module.
            If you have implemented custom tokenizer, add it's import to tacotron2.tokenizers.__init__.py file
        :param load_mel_from_disk:
        :param max_wav_value:
        :param sampling_rate:
        :param filter_length:
        :param hop_length:
        :param win_length:
        :param n_mel_channels:
        :param mel_fmin:
        :param mel_fmax:
        :param n_frames_per_step:
        """

        self.audiopaths_and_text = load_filepaths_and_text(meta_file_path)
        self.tokenizer = Factory.get_object(f'tacotron2.tokenizers.{tokenizer_class_name}')

        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.n_frames_per_step = n_frames_per_step
        self.load_mel_from_disk = load_mel_from_disk

        self.stft = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax
        )

    @classmethod
    def from_hparams(cls, hparams: HParams, is_valid: bool) -> 'TextMelLoader':
        """Build class instance from hparams map
        If you create dataset instance via this method, make sure, that meta_train.txt (if is_valid==False) or
            meta_valid.txt (is is_valid==True) exists in the dataset directory
        :param hparams: HParams, dictionary with parameters
        :param is_valid: bool, get validation dataset or not (train)
        :return: TextMelLoader, dataset instance
        """
        param_names = inspect.getfullargspec(TextMelLoader.__init__).args
        params = dict()
        for param_name in param_names:
            if param_name == 'self':
                continue
            elif param_name == 'meta_file_path':
                data_directory = Path(hparams.data_directory)
                postfix = 'valid' if is_valid else 'train'
                value = data_directory / f'meta_{postfix}.txt'
                if not value.is_file():
                    raise FileNotFoundError(f"Can't find {str(value)} file. Make sure, that file exists")
            else:
                value = hparams[param_name]

            params[param_name] = value

        obj = TextMelLoader(**params)
        return obj

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text: str):
        text_norm = torch.IntTensor(self.tokenizer.encode(text))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

    @staticmethod
    def get_collate_function(pad_id, n_frames_per_step):
        def collate(batch):
            """Collate's training batch from normalized text and mel-spectrogram"""
            # Right zero-pad all one-hot text sequences to max input length
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x[0]) for x in batch]),
                dim=0, descending=True)
            max_input_len = input_lengths[0]

            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.fill_(pad_id)
            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]][0]
                text_padded[i, :text.size(0)] = text

            # Right zero-pad mel-spec
            num_mels = batch[0][1].size(0)
            max_target_len = max([x[1].size(1) for x in batch])
            if max_target_len % n_frames_per_step != 0:
                max_target_len += n_frames_per_step - max_target_len % n_frames_per_step
                assert max_target_len % n_frames_per_step == 0

            # include mel padded and gate padded
            mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
            mel_padded.zero_()
            gate_padded = torch.FloatTensor(len(batch), max_target_len)
            gate_padded.zero_()
            output_lengths = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                mel = batch[ids_sorted_decreasing[i]][1]
                mel_padded[i, :, :mel.size(1)] = mel
                gate_padded[i, mel.size(1) - 1:] = 1
                output_lengths[i] = mel.size(1)

            return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

        return collate

    def get_data_loader(self, batch_size: int, is_distributed: bool, shuffle: bool):
        """Construct DataLoader object from the Dataset object

        :param is_distributed: bool, set distributed sampler or not
        :param batch_size: int, batch size
        :param shuffle: bool, shuffle data or not
        :return: DataLoader
        """

        sampler = DistributedSampler(self, shuffle=shuffle) if is_distributed else None
        shuffle = shuffle if sampler is None else False

        collate_fn = self.get_collate_function(pad_id=self.tokenizer.pad_id, n_frames_per_step=self.n_frames_per_step)
        dataloader = DataLoader(
            self,
            num_workers=1,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=shuffle
        )

        return dataloader
