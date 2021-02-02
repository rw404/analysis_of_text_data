from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk

class KMeansClusters(BaseEstimator, TransformerMixin):

    def __init__(self, k = 7):
        """
        k - число кластеров, model - реализация KMeans
        """
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters = True)

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        """
        Обучение модели векторными представлениями, полученными прямым кодированием 
        """
        return self.model.cluster(documents, assign_clusters = True)


"""
Из TextNormalizer:
def transform(self, documents):
    return [' '.join(self.normalize(doc)) for doc in documents]
"""

# Векторизация
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class OneHotVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.vectorizer = CountVectorizer(binary = True)

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        freqs = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in freqs]

# Pipes
from sklearn.pipeline import Pipeline

corpus = PickledCorpusReader('../corpus')
docs = corpus.docs(categories = ['news'])

model = Pipeline([
    ('norm', TextNormalizer()),
    ('vect', OneHotVectorizer()),
    ('clusters', KMeansClusters(k = 7))
])

clusters = model.fit_transform(docs)
pickles = list(corpus.fileids(categories = ['news']))
for idx, cluster in enumerate(clusters):
    print("Document '{}' assigned to cluster {}.".format(pickles[idx], cluster))

# Оптимизация

from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin

class KMeansClusters(BaseEstimator, TransformerMixin):

    def __init__(self, k = 7):
        self.k = k
        self.model = MiniBatchKMeans(self.k)

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        return self.model.fit_predict(documents)

