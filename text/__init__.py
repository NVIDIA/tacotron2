""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import eng_symbols, kor_symbols
from hparams import create_hparams

hparam = create_hparams()
cleaner_names = hparam.text_cleaners

# Mappings from symbol to numeric ID and vice versa:
symbols = ""
_symbol_to_id = {}
_id_to_symbol = {}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def change_symbol(cleaner_names):
  symbols = ""
  global _symbol_to_id
  global _id_to_symbol
  if cleaner_names == ["english_cleaners"]: symbols = eng_symbols
  if cleaner_names == ["korean_cleaners"]: symbols = kor_symbols

  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  _id_to_symbol = {i: s for i, s in enumerate(symbols)}

change_symbol(cleaner_names)

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  change_symbol(cleaner_names)
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    try:
      if not m:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
        break
      sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
      sequence += _arpabet_to_sequence(m.group(2))
      text = m.group(3)
    except:
      print(text)
      exit()
  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

if __name__ == "__main__":
  print(text_to_sequence('this is test sentence.? ', ['english_cleaners']))
  print(text_to_sequence('테스트 문장입니다.? ', ['korean_cleaners']))
  print(_clean_text('AB테스트 문장입니다.? ', ['korean_cleaners']))
  print(_clean_text('mp3 파일을 홈페이지에서 다운로드 받으시기 바랍니다.',['korean_cleaners']))
  print(_clean_text("마가렛 대처의 별명은 '철의 여인'이었다.", ['korean_cleaners']))
  print(_clean_text("제 전화번호는 01012345678이에요.", ['korean_cleaners']))
  print(_clean_text("‘아줌마’는 결혼한 여자를 뜻한다.", ['korean_cleaners']))
  print(text_to_sequence("‘아줌마’는 결혼한 여자를 뜻한다.", ['korean_cleaners']))

