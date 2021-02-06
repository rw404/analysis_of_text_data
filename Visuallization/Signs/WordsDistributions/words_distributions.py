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

from yellowbrick.text.freqdist import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer

hobby_types = {}

for category in corpus['categories']:
    texts = []
    for idx in range(len(corpus['data'])):
        if corpus['target'][idx] == category:
            texts.append(corpus['data'][idx])
    hobby_types[category] = texts

# cooking
vectorizer = CountVectorizer(stop_words = 'english')
docs = vectorizer.fit_transform(text for text in hobby_types['cooking'])
features = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features = features)
visualizer.fit(docs)
visualizer.poof(outpath = "./cooking.png")

# gaming
vectorizer = CountVectorizer(stop_words = 'english')
docs = vectorizer.fit_transform(text for text in hobby_types['gaming'])
features = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features = features)
visualizer.fit(docs)
visualizer.poof(outpath = "./gaming.png")


