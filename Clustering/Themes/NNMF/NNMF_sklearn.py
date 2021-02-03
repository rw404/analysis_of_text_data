from sklearn.decomposition import NMF

class SklearnTopicModels(object):

    def __init__(self, n_topics = 50, estimator = 'LDA'):
        """
        n_topics -- кол-во тем
        estimator =
            'LDA' - латентное размещение Дирихле ('LDA')
            'LSA' - латентно-семантический анализ
            'NMF' - неторицательное матричное разложение
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components = self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components = self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_topics = self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('tfidf', CountVectorizer(tokenizer = identity,
                                      preprocessor = None, lowercase = False)),
            ('model', self.estimator)
        ])
