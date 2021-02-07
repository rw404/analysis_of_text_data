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

from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer

"""
# SilhouetteVisualizer
# Создание и проверка модели кластеризации и визуализации
visualizer = SilhouetteVisualizer(KMeans(n_clusters = 6))
visualizer.fit(docs)
visualizer.poof('Silhoutte.png')
"""

from yellowbrick.cluster import KElbowVisualizer

# Локтевые кривые
# Создание визуализаций для разного числа кластеров

visualizer = KElbowVisualizer(KMeans(), metric = 'silhouette', k = [3, 19])
visualizer.fit(docs)
visualizer.poof('KElbows.png')
