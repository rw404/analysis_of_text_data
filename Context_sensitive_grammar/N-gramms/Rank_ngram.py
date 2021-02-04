from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures

def rank_quadgrams(corpus, metric, path = None):
    """
    Находит и оценивает тетраграммы в файл, если указан, иначе возвращает список в памяти.
    """
    # Создать объект оценки словосочетаний из слов в корпусе.
    ngrams = QuadgramCollocationFinder.from_words(corpus.words())

    # Оценить словосочетания в соответствии с заданной метрикой
    scored = ngrams.score_ngrams(metric)

    if path:
        # Результаты в файл
        with open(path, 'w') as f:
            f.write("Collocation\tScore ({})".format(metric.__name__))
            for ngram, score in scored:
                f.write("{}\t{}\n".format(repr(ngram), score))
    else:
        return scored

rank_quadgrams(
    corpus, QuadgramAssocMeasures.likelihood_ratio, 'quadgrams.txt'
)

from sklearn.base import BaseEstimator, TransformerMixin

class SignificantCollocations(BaseEstimator, TransformerMixin):
    """
    Значимые словосочетания
    """
    def __init__(self,
                 ngram_class = QuadgramCollocationFinder,
                 metric = QuadgramAssocMeasures.pmi):
        self.ngram_class = ngram_class
        self.metric = metric

    def fit(self, docs, target):
        ngrams = self.ngram_class.from_documents(docs)
        self.scored_ = dict(ngrams.score_ngrams(self.metric))

    def transform(self, docs):
        for doc in docs:
            ngrams = self.ngram_class.from_words(docs)
            yield {
                ngram: self.scored_.get(ngram, 0.0)
                for ngram in ngrams.nbest(QuadgramAssocMeasures.raw_freq, 50)
            }

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import TfidfVectorizer

model = Pipeline([
    ('union', FeatureUnion(
        transformer_list = [
            ('ngrams', Pipeline([
                ('sigcol', SignificantCollocations()),
                ('dsigcol', DictVectorizer()),
            ])),

            ('tfidf', TfidfVectorizer()),
        ]
    )),

    ('clf', SGDClassifier()),
])
