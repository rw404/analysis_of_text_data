from gensim.models.doc2vec import TaggedDocument, Doc2Vec

corpus = [list(tokenize(doc)) for doc in corpus]
corpus = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
        ]

model = Doc2Vec(corpus, size = 5, min_count = 0)
print(model.docvecs[0])
