def vectorize(doc):
    return {
            token: True
            for token in doc
            }

vectors = map(vectorize, corpus)

