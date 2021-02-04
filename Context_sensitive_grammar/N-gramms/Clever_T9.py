from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist

from collections import defaultdict

# Символы дополнения неопределенных элементов
UNKNOWN = "<UNK>"
LPAD = "<s>"
RPAD = "</s>"

class NgramCounter(object):
    """
    Класс NgramCounter подсчитывает n-граммы для заданного словаря и размера окна.
    """
    def __init__(self, n, vocabulary, unknown = UNKNOWN):
        """
        n -- размер n-грамм
        """
        if n < 1:
            raise ValueError("ngram size must be greater than or equal to 1")

        self.n = n
        self.unknown = unknown
        self.padding = {
            "pad_left": True,
            "pad_right": True,
            "left_pad_symbol": LPAD,
            "right_pad_symbol": RPAD,
        }

        self.vocabulary = vocabulary
        self.allgrams = defaultdict(ConditionalFreqDist)
        self.ngrams = FreqDist()
        self.unigrams = FreqDist()

    def train_counts(self, trraining_text):
        for sent in trraining_text:
            checked_sent = (self.che_against_vocab(word) for word in sent)
            sent_start = True
            for ngram in self.to_ngrams(checked_sent):
                self.ngrams[ngram] += 1
                context, word = tuple(ngram[:-1]), ngram[-1]
                if sent_start:
                    for context_word in context:
                        self.unigrams[context_word] += 1
                    sent_start = False

                for window, ngram_order in enumerate(range(self.n, 1, -1)):
                    context = context[window:]
                    self.allgrams[ngram_order][context][word] += 1
                self.unigrams[word] += 1

    def check_against_vocab(self, word):
        if word in self.vocabulary:
            return word
        return self.unknown

    def to_ngrams(self, sequence):
        """
        Обертка для метода ngrams из библиотеки NLTK
        """
        return ngrams(sequence, self.n, **self.padding)

# Частоты n-грамм
def count_ngrams(n, vocabulary, texts):
    counter = NgramCounter(n, vocabulary)
    counter.train_counts(texts)
    return counter

if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')
    tokens = [''.join(word[0]) for word in corpus.words()]
    vocab = Counter(tokens)
    sents = list([word[0] for word in sent] for sent in corpus.sents())
    trigram_counts = count_ngrams(3, covab, sents)

# Распределение частот
print(trigram_counts.unigrams)

# Для n-грам более высокого порядка
print(trigram_counts.ngrams[3])

# Возможные предшествующие контексты
print(sorted(trigram_counts.ngrams[3].conditions()))

# Возможный список последующих слов
print(list(trigram_counts.ngrams[3][('the', 'President')]))
