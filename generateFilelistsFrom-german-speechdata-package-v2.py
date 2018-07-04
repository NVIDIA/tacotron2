import os
from lxml import etree
from scipy.io.wavfile import read

for tsk in ('train', 'test'):

    path = f'/Volumes/Trainings/datasets/german-speechdata-package-v2/{tsk}'
    filepath = f'/Users/hendorf/code/audiodrama/project/tacotron2/filelists/german-speechdata-package-v2_{tsk}_filelist.txt'

    counter = 0
    with open(filepath, 'w', 128) as f:
        for audio in os.listdir(path):

            if '_Yamaha' not in audio:
                continue

            try:
                sampling_rate, data = read(os.path.join(path, audio))
            except:
                print(f'corrupted audio file: {audio}')
                continue

            if os.path.getsize(os.path.join(path, audio)) > 600 * 1024:
                continue

            tree = etree.parse(os.path.join(path, f'{audio.split("_")[0]}.xml'))
            if tree.xpath('/recording/gender[1]/text()')[0] != 'male':
                continue
            if tree.xpath('/recording/muttersprachler[1]/text()')[0].lower() != 'ja':
                continue
            text = tree.xpath('/recording/cleaned_sentence[1]/text()')[0]
            text = text.replace("\n", "")

            f.write(f'{os.path.join(path, audio)}|{text}\n')
            counter += 1

    print(counter)
