from sklearn.base import TransformerMixin

class Transformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        """
        Правила преобразования на основе входных данных X
        """
        return self

    def transform(self, X):
        """
        Преобразует X в новый набор данных Xprime и возвращает его
        """
        return Xprime

