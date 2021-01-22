import gensim

corpus = [tokenize(doc) for doc in corpus]
id2word = gensim.corpora.Dictionary(corpus)
vectors = [
        id2word.doc2bow(doc) for doc in corpus
        ]
