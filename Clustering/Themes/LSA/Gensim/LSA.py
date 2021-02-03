from gensim.sklearn_api import lsimodel, ldamodel

class GensimTopicModels(object):

    def __init__(self, n_topics = 50, estimator = 'LDA'):
        """
        n_topics - кол-во тем

        estimator либо латентное размещение Дирихле ('LDA'), либо латентно-семантический анализ 'LSA'
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = lsimodel.LsiTransformer(num_topics = self.n_topics)
        else:
            self.extimator = ldamodel.LdaTransformer(num_topics = self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', GensimTdidfVectorizer()),
            ('model', self.estimator)
        ])
