import nltk
from functools import partial

LPAD_SYMBOL = "<s>"
RPAD_SYMBOL = "</s>"

nltk_ngrams = partial(
    nltk.ngrams,
    pad_right = True, right_pad_symbol = RPAD_SYMBOL,
    left_pad = True, left_pad_symbol = LPAD_SYMBOL
)

    def ngrams(self, n = 2, fileids = None, categories = None):
        for sent in self.sents(fileids = fileids, categories = categories):
            for ngram in nltk.ngrams(sent, n):
                yield ngram


