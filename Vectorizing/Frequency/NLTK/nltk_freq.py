from collections import defaultdict

def vectorize(doc):
    """
    Применять только к объекту класса Preprocessor
    """
    # Изначально слварь пуст и хранит нули
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features

vectors = map(vectorize, corpus)
