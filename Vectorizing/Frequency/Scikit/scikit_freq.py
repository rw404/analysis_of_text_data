from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectros = vectorizer.fit_transform(corpus)
