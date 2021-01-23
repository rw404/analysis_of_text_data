import os
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full

class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path = None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels = None):
        self.id2word = Dictionary(documents)
        self.save()
        return self

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))

# Нормализация

import unicodedata
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language = 'english'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
                unicodedata.category(char).startswith('P') for char in torken
                )

    def is_stopword(self, token):
        return token.lower() in self.stopwords
    
    def normalize(self, document):
        return [
                self.lemmatize(token, tag).lower()
                for paragraph in document
                for sentence in paragraph
                for (token, tag) in sentence
                if not self.is_punct(token) and not self.is_stopword(token)
                ]

    def lemmatize(self, token, pos_tag):
        """
        Преобразуем теги частей речи из набора Penn Treebank функции nltk.pos_tag
        в теги WordNet
        """
        tag = {
                'N': wn.NOUN,
                'V': wn.VERB,
                'R': wn.ADV,
                'J': wn.ADJ
                }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y = None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)
