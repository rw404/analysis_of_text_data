from nltk.text import TextCollection

def vectroize(corpus):
    corpus = [tokenize(doc) for doc in corpus]
    texts = TextCollection(corpus)

    for doc in corpus:
        yield {
                term: texts.tf_idf(term, doc)
                for term in doc
                }


