class GensimTfidfVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, dirpath = ".", tofull = False):
        """
        Путь к каталогу с лексиком в файле corpus.dict
        и моделью TF-IDF в файле tfidf.model.

        Передаем tofull = True, если следующим этапом является объект
        Estimator из Scikit-Learn, иначе, если следующим этапом является
        модель Gensim, оставьте значение False.
        """
        self._lexicon_path = os.path.join(dirpath, "corpus.dict")
        self._tfidf_path = os.path.join(dirpath, "tfidf.model")

        self.lexicon = None
        self.tfidf = None
        self.tofull = tofull

        self.load()

    def load(self):
        if os.path.exists(self._lexicon_path):
            self.lexicon = Dictionary.load(self._lexicon_path)

        if os.path.exists(self._tfidf_path):
            self.tfidf = TfidfModel().load(self._tfidf_path)

    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.tfidf.save(self._tfidf_path)

    def fit(self, documetns, labels = None):
        self.lexicon = Dictionary(documents)
        self.tfidf = TfidfModel([
            self.lexicon.doc2bow(doc)
            for doc in documents
        ],
                                id2word = self.lexicon)
        self.save()
        return self

    def transform(self, documents):
        def generator():
            for document in documents:
                vec = self.tfidf[self.lexicon.doc2bow(document)]
                if self.tofull:
                    yield sparse2full(vec)
                else:
                    yield vec
        return list(generator())

from sklearn.pipeline import Pipeline
from gensim.sklearn_api import ldamodel

class GensimTopicModels(object):

    def __init__(self, n_topics = 50):
        """
        n_topics - число тем
        """
        self.n_topics = n_topics
        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', GensimTfidfVectorizer()),
            ('model', ldamodel.LdaTransformer(num_topics = self.n_topics))
        ])

    def fit(self, documents):
        self.model.fit(documents)

        return self.model

if __name__ == '__main__':
    corpus = PickledCorpusReader('../corpus')

    gensim_lda = GensimTopicModels()

    docs = [
        list(corpus.docs(fileids = fileid))[0]
        for fileid in corpus.fileids()
    ]

    gensim_lda.fit(docs)

lda = gensim_lda.model.named_steps['model'].gensim_model
print(lda.show_topics())

# Добавление специальной функции получения тем для предыдущего класса
    def get_topics(vectorized_corpus, model):
        from operator import itemgetter

        topics = [
            max(model[doc], key = itemgetter(1))[0]
            for doc in vectorized_corpus
        ]

        return topics

    lda = gensim_lda.model.named_steps['model'].gensim_model

    corpus = [
        gensim_lda.model.named_steps['vect'].lexicon.doc2bow(doc)
        for doc in gensim_lda.model.named_steps['norm'].transform(docs)
    ]

    topics = get_topics(corpus, lda)

    for topic, doc in zip(topics, docs):
        print("Topic:{}".format(topic))
        print(doc)
