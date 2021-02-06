from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer


# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

import os
import yellowbrick as yb
from sklearn.datasets.base import Bunch

## Путь к тектам
FIXTURES = os.path.join(os.getcwd(), "data")

## Механизм загрузки корпуса
corpora = {
    "hobbies": os.path.join(FIXTURES, "hobbies")
}

def load_corpus(name):
    """
    Загрузка и извелечение корпуса
    """

    # Путь из наборов данных
    path = corpora[name]

    # Каталоги в каталоге как категории
    categories = [
        cat for cat in os.listdir(path)
        if os.path.isdir(os.path.join(path, cat))
    ]

    files = [] # список имен файлов
    data = [] # текст из файла
    target = [] # строка с названием категории

    # Загрузить данные из файла в корпус
    for cat in categories:
        for name in os.listdir(os.path.join(path, cat)):
            files.append(os.path.join(path, cat, name))
            target.append(cat)

            with open(os.path.join(path, cat, name), 'r') as f:
                data.append(f.read())

    # Вернуть пакет данных для использования, как в примере с новостями
    return Bunch(
        categories = categories,
        files = files,
        data = data,
        target = target,
    )

corpus = load_corpus('hobbies')

tfidf = TfidfVectorizer()
docs = tfidf.fit_transform(corpus.data)

tsne = TSNEVisualizer()
tsne.fit(docs)
tsne.poof('./simple_view.png')

# Кластеризация
from sklearn.cluster import KMeans

clusters = KMeans(n_clusters = 5)
clusters.fit(docs)
tsne = TSNEVisualizer()
tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
tsne.poof('./clustered_view.png')

# Классификация
labels = corpus.target

tsne = TSNEVisualizer()
tsne.fit(docs, labels)
tsne.poof('./classification.png')
