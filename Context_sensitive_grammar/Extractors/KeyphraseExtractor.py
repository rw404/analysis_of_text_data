from sklearn.base import BaseEstimator, TransformerMixin
from itertools import groupby
from nltk.chunk import tree2conlltags
from unicodedata import category as unicat
GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
GOODTAGS = frozenset(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])

class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    """
    Обертывает PickledCorpusReader, содержащий маркированные документы.
    """
    def __init__(self, grammar = GRAMMAR):
        self.grammar = GRAMMAR
        self.chunker = RegexpParser(self.grammar)

    def normalize(self, sent):
        """
        Удаляет знаки препинания из лексемизированного/маркированного предложения и преобразует
        буквы в нижний регистр.
        """
        is_punct = lambda word: all(unicat(c).startswith('P') for c in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_keyphrases(self, document):
        """
        Выполняет парсинг предложений из документа, используя парсер с грамматикой, преобразует
        дерево синтаксического анализа в маркированную последовательность.
        Возвращает извлеченные фразы.
        """
        for sents in document:
            for sent in sents:
                sent = self.normalize(sent)
                if not sent: continue
                chunks = tree2conlltags(self.chunker.parse(sent))
                phrases = [
                    " ".join(word for word, pos, chunk in group).lower()
                    for key, group in groupby(
                        chunks, lambda term:term[-1] != 'O'
                    ) if key
                ]
                for phrase in prases:
                    yield phrase

    def fit(self, documents, y = None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.extract_keyphrases(document)


