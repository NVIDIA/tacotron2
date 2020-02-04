from tacotron2.tokenizers._grapheme_tokenizer import GraphemeTokenizer
from tacotron2.tokenizers._utilities import replace_numbers_with_text, clean_spaces


class RussianGraphemeTokenizer(GraphemeTokenizer):
    """Russian graphemes-lvl tokenizer"""

    def __init__(self):
        language = 'ru'
        letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        super().__init__(language, letters)

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = replace_numbers_with_text(text, self.language)
        text = clean_spaces(text)
        return text
