import re
from pathlib import Path
from typing import List, Dict

from pandas.core.common import flatten
from russian_g2p.Transcription import Transcription
from russian_g2p.modes.Phonetics import Phonetics

from tacotron2.tokenizers._tokenizer import Tokenizer
from tacotron2.tokenizers._utilities import replace_numbers_with_text, clean_spaces


class RussianPhonemeTokenizer(Tokenizer):
    """Russian phonemes-lvl tokenizer
    It uses pre-calculated phonemes dictionary. If some specific word is not in the dictionary, then the
    russian_g2p.Transcription will be applied (https://github.com/nsu-ai/russian_g2p)
    """

    @property
    def pad_id(self):
        return 0

    @property
    def unk_id(self):
        return 1

    def __init__(self):
        self.russian_phonemes = sorted(Phonetics().russian_phonemes_set)
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.id2token = [self.pad, self.unk] + self.russian_phonemes + [' ']
        self.token2id = {token: id_ for id_, token in enumerate(self.id2token)}

        self.word2phonemes = self.read_phonemes_corpus(Path(__file__).parent / 'data/russian_phonemes_corpus.txt')
        self.word_regexp = re.compile(r'[А-яЁё]+')
        self.transcriptor = Transcription()

    @staticmethod
    def read_phonemes_corpus(file_path: Path) -> Dict[str, List[str]]:
        """Read pre-calculated phonemes corpus (word to phonemes list map)
        :param file_path: Path, path to the corpus file
        :return: dict, word to phonemes dictionary
        """
        phonemes_corpus = dict()
        with file_path.open() as f:
            for line in f.readlines():
                line_split = line.strip().split()
                phonemes_corpus[line_split[0].lower()] = line_split[1:]

        return phonemes_corpus

    def encode(self, text: str) -> List[int]:
        """Tokenize and encode text on phonemes-lvl
        :param text: str, input text
        :return: list, of phonemes ids
        """
        text = text.lower()
        text = replace_numbers_with_text(text, lang='ru')
        text = clean_spaces(text)

        tokens = self._tokenize(text)
        token_ids = [self.token2id[token] for token in tokens]
        return token_ids

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text on phonemes. Uses dictionary if word is presented, or calculate phonemes and add new word to
        dictionary (to not to calculate next time)
        :param text: str, input text
        :return: list, of phonemes
        """
        tokens = []

        word_matches = list(self.word_regexp.finditer(text))
        for i_word_match, word_match in enumerate(word_matches):
            matched_word = word_match.group(0)
            matched_word_tokens = self.word2phonemes.get(matched_word, None)

            if matched_word_tokens is None:
                matched_word_tokens = flatten(self.transcriptor.transcribe([matched_word]))
                self.word2phonemes[matched_word] = matched_word_tokens

            if i_word_match != len(word_matches) - 1:
                matched_word_tokens.append(' ')

            tokens.extend(matched_word_tokens)

        return tokens
