from sklearn.base import BaseEstimator

class Estimator(BaseEstimator):

    def fit(self, X, y = None):
        """
        Возвращает self
        """
        return self

    def predict(self, X):
        """
        Входные данные X и возвращает вектор предсказаний для каждой строки
        """
        return yhat

# Пример с байесовской моделью мультиноального распределения
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha = 0.0, class_prior = [0.4, 0.6])
model.fit(document, labels)
