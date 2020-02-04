from abc import ABC, abstractmethod, abstractproperty
from typing import List


class Tokenizer(ABC):
    """Abstract tokenizer class from which all tokenizers must be inherited"""

    @abstractproperty
    def pad_id(self):
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    def encode_many(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]
