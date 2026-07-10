import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import BPEDecoder

class SocrateXTokenizer:
    """
    Wrapper over HuggingFace Tokenizers to provide a clean interface:
    .fit(), .encode(), .decode(), .save()
    """
    def __init__(self, tokenizer=None, vocab_size=1000):
        if tokenizer is None:
            self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self._tokenizer.pre_tokenizer = Whitespace()
            self._tokenizer.decoder = BPEDecoder()
        else:
            self._tokenizer = tokenizer
            if self._tokenizer.decoder is None:
                self._tokenizer.decoder = BPEDecoder()
        self.vocab_size = vocab_size

    def fit(self, data_source, special_tokens=None):
        """
        Trains the tokenizer.
        data_source can be a path to a text file (e.g. "data.txt")
        or a list of strings in memory.
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
            
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
        
        if isinstance(data_source, str) and os.path.exists(data_source):
            self._tokenizer.train(files=[data_source], trainer=trainer)
        elif isinstance(data_source, list):
            self._tokenizer.train_from_iterator(data_source, trainer=trainer)
        else:
            raise ValueError("data_source must be a list of strings or a path to a text file.")
            
    def encode(self, text):
        return self._tokenizer.encode(text)
        
    def decode(self, ids):
        return self._tokenizer.decode(ids)
        
    def get_vocab_size(self):
        return self._tokenizer.get_vocab_size()
        
    def token_to_id(self, token):
        return self._tokenizer.token_to_id(token)
        
    def save(self, path):
        self._tokenizer.save(path)
        
    @classmethod
    def from_file(cls, path):
        tk = Tokenizer.from_file(path)
        return cls(tokenizer=tk, vocab_size=tk.get_vocab_size())

def init_tokenizer(vocab_size=1000):
    """
    Factory function to quickly instantiate an empty tokenizer,
    ready to be trained via .fit()
    """
    return SocrateXTokenizer(vocab_size=vocab_size)
