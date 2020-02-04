import re

from num2words import num2words


def replace_numbers_with_text(text: str, lang: str) -> str:
    """Convert all numbers in text to theirs words representation

    :param text: str, input text
    :param lang: str, text language (check num2words lib.)
    :return: str, output string with replaced numbers
    """

    number_regexp = re.compile(r'\d*\.?\d+', )

    matched_numbers = []

    for digit_match in number_regexp.finditer(text):
        matched_number = num2words(digit_match.group(0), lang=lang, )
        matched_numbers.append(matched_number)

    text_split = number_regexp.split(text)
    new_text_substrings = []

    for i in range(len(text_split)):
        new_text_substrings.append(text_split[i])
        if i < len(text_split) - 1:
            new_text_substrings.append(matched_numbers[i])

    text = ''.join(new_text_substrings)
    return text


def clean_spaces(text: str) -> str:
    """Clean spaces in text (replace all space sequences with single space)
    :param text: str, input text
    :return: str, output text
    """
    return re.sub(' +', ' ', text)
