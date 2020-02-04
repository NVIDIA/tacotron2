from unidecode import unidecode

from tacotron2.tokenizers._english_abbreviations_utilities import expand_abbreviations
from tacotron2.tokenizers._english_numbers_utilities import normalize_numbers
from tacotron2.tokenizers._grapheme_tokenizer import GraphemeTokenizer
from tacotron2.tokenizers._utilities import clean_spaces


class EnglishGraphemeTokenizer(GraphemeTokenizer):
    """English graphemes-lvl tokenizer"""

    def __init__(self):
        language = 'en'
        letters = 'abcdefghijklmnopqrstuvwxyz'
        super().__init__(language, letters)

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = unidecode(text)
        text = normalize_numbers(text)
        text = expand_abbreviations(text)
        text = clean_spaces(text)

        return text