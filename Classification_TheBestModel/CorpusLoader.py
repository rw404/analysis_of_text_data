from sklearn.model_selection import KFold

class CorpusLoader(object):

    def __init__(self, reader, folds = 12, shuffle = True, categories = None):
        self.reader = reader
        self.folds = KFold(n_splits = folds, shuffle = shuffle)
        self.files = np.asarray(self.reader.fileids(categories = categories))

    def fileids(self, idx = None):
        if idx is None:
            return self.files
        return self.files[idx]

    def documents(self, idx = None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids = [fileid]))

    def labels(self, idx = None):
        return [
            self.reader.categories(fileids = [fileid])[0]
            for fileid in self.fileids(idx)
        ]

    def __iter__(self):
        for train_index, test_index in self.folds.split(self.files):
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)

            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test


from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def identity(words):
    return words

def create_pipeline(estimator, reduction = False):

    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer = identity, preprocessor = None, lowercase = False
        ))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components = 10000)
        ))

    # Объект оценки
    steps.append(('classifier', estimator))
    return Pipeline(steps)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

models = []
for form in (LogisticRegression, MultinomialNB, SGDClassifier):
    models.append(create_pipeline(form(), True))
    models.append(create_pipeline(form(), False))

for model in models:
    model.fir(train_docs, train_labels)

# Оценка Моделей
import numpy as np

from sklearn.metrics import accuracy_score

for model in models:
    scores = [] # Список оценок
    
    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    print("Accuracy of {} is {:0.3f}".format(model, np.mean(scores)))

# Отчет о модели
from sklearn.metrics import classification_report

model = create_pipeline(SGDClassifier(), False)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, labels = labels))

# Выбор наилучшей модели по таблицам
import tabulate
import numpy as np

from collections import defaultdict
from seklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

fields = ['model', 'precision', 'recall', 'accuracy', 'f1']
table = []

for model in models:
    scores = defaultdict(list) # Оценки текущей модели

    # Перекрестная проверка
    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Оценки в score
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred))

    # Группировка оценок и загрузка в таблицу
    row = [str(model)]
    for field in fields[1:]:
        row.append(np.mean(scores[field]))

    table.append(row)

# Сортировка таблицы по f1-score и вывод
table.sort(key = lambda row: row[-1], reverse = True)
print(tabulate.tabulate(table, headers = fields))

# Эксплуатация модели
import pickle
from datetime import datetime

time = datetime.now().strftime("%Y-%m-%d")
path = 'hobby-classifier-{}'.format(time)

with open(path, 'wb') as f:
    pickle.dump(model, f)

# Загрузка и использование
import nltk
def preprocess(text):
    return [
        [
            list(nltk.pos_tag(nltk.word_tokenize(sent)))
            for sent in nltk.sent_tokenize(para)
        ] for parain text.split("\n\n")
    ]

with open(path, 'rb') as f:
    model.predict([preprocess(doc) for doc in newdocs])
