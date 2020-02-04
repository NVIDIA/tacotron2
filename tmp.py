from pathlib import Path

from tacotron2.hparams import HParams
from tacotron2.model import Tacotron2
from tacotron2.utils import to_device

hparams = HParams.from_yaml(Path('../configs/hparams.default.json'))

if __name__ == '__main__':
    device = 'cuda'
    hparams.training_files = './data/LJSpeech-1.1/meta_train.txt'
    hparams.validation_files = './data/LJSpeech-1.1/meta_valid.txt'
    hparams.batch_size = 2

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    tacotron = Tacotron2(hparams=hparams).to(device)

    for batch in train_loader:
        batch = to_device(list(batch), device=device)
        res = tacotron(batch)
        print(res)
