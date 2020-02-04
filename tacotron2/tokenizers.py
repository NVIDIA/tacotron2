import re
from abc import ABC, abstractmethod,abstractproperty
from string import punctuation
from typing import List

from num2words import num2words


class Tokenizer(ABC):

    @abstractproperty
    def pad_id(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    def encode_many(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]


class RussianGraphemeTokenizer(Tokenizer):
    """Russian graphemes-lvl tokenizer

    Tokenizer which encodes russian text string into a sequence of grapheme indexes. It also allows to replace
    numbers with their word-representations
    """

    NUMBER_REGEXP = re.compile(r'\d*\.?\d+', )

    ALPHABET = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя" + punctuation + ' '
    PAD = '<PAD>'
    UNK = '<UNK>'
    ID2TOKEN = [PAD, UNK] + list(ALPHABET)
    TOKEN2ID = {token: id_ for id_, token in enumerate(ID2TOKEN)}
    unk_id = TOKEN2ID[UNK]
    pad_id = TOKEN2ID[PAD]

    def __init__(self, replace_numbers_with_text: bool = True):
        """
        :param replace_numbers_with_text: bool, do numbers to text conversion flag
        """
        self.replace_numbers_with_text = replace_numbers_with_text

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token (graphemes) indexes

        :param text: str, input text
        :return: list, grapheme indexes
        """
        text = text.lower()
        if self.replace_numbers_with_text:
            text = RussianGraphemeTokenizer._replace_numbers_with_text(text, lang='ru')

        token_ids = RussianGraphemeTokenizer._numericalize(text)

        return token_ids

    @staticmethod
    def _numericalize(text: str) -> List[int]:
        token_ids = [RussianGraphemeTokenizer.TOKEN2ID.get(token, RussianGraphemeTokenizer.unk_id) for token in text]
        return token_ids

    @staticmethod
    def _replace_numbers_with_text(text: str, lang: str) -> str:
        """Convert all numbers in text to theirs words representation

        :param text: str, input text
        :param lang: str, text language (check num2words lib.)
        :return: str, output string with replaced numbers
        """
        matched_numbers = []

        for digit_match in RussianGraphemeTokenizer.NUMBER_REGEXP.finditer(text):
            matched_number = num2words(digit_match.group(0), lang=lang, )
            matched_numbers.append(matched_number)

        text_split = RussianGraphemeTokenizer.NUMBER_REGEXP.split(text)
        new_text_substrings = []

        for i in range(len(text_split)):
            new_text_substrings.append(text_split[i])
            if i < len(text_split) - 1:
                new_text_substrings.append(matched_numbers[i])

        text = ''.join(new_text_substrings)
        return text