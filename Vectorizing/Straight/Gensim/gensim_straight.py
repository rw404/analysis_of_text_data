import gensim

corpus = [tokenize(doc) for doc in corpus]
id2word = gensim.corpora.Dictionary(corpus)
vectors = [
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in corpus
        ]
