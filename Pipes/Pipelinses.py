from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

model = Pipeline([
    ('normalizer', TextNormalizer()),
    ('vectorizer', GensimVectorizer()),
    ('bayes', MultinomialNB()),
    ])

# Поиск по сетке
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(model, param_grid = {
    'count__analyzer': ['words', 'char', 'char_wb'], # определяет для CountVectorizer возможные
                                                     # значения: n-граммы после слов, после
                                                     # символов, по символам между границами слов
    'count__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3)],
    'onehot__threshold': [0.0, 1.0, 2.0, 3.0],
    'bayes__alpha': [0.0, 1.0],
    })

