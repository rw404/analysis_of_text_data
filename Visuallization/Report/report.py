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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from sklearn.linear_model import SGDClassifier

tfidf = TfidfVectorizer()
docs = tfidf.fit_transform(corpus.data)
labels = corpus.target

X_train, X_test, y_train, y_test = train_test_split(
    docs.toarray(), labels, test_size = 0.2
)

"""
# Gaussian_model
visualizer = ClassificationReport(GaussianNB(), classes = corpus.categories)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

visualizer.poof("Gaussian_report.png")
"""

"""
# SGDClassifier
visualizer1 = ClassificationReport(SGDClassifier(), classes = corpus.categories)

visualizer1.fit(X_train, y_train)
visualizer1.score(X_test, y_test)

visualizer1.poof("SGD_report.png")

"""

from yellowbrick.classifier import ConfusionMatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

"""
# LogisticRegression
visualizer = ConfusionMatrix(LogisticRegression(), classes = corpus.categories)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof("ConfusMatrix_LogisticRegression.png")

"""
# MultinomialNB
visualizer2 = ConfusionMatrix(MultinomialNB(), classes = corpus.categories)

visualizer2.fit(X_train, y_train)
visualizer2.score(X_test, y_test)
visualizer2.poof("ConfusMatrix_MultinomialNB.png")

