from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import CountVectorizer

class SklearnTopicModels(object):

    def __init__(self, n_topics = 50, estimator = 'LDA'):
        """
        n_topics - число тем
        Для латентно-сумантического анализа estimator = 'LSA', для латентного размещения Дирихле
        ('LDA').
        """
        self.n_topics = n_topics
        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components = self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_topics = self.n_topics)
        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', CountVectorizer(tokenizer = identity,
                                     preprocessor = None, lowercase = False)),
            ('model', self.estimator),
        ])

    def fit_transform(self, documents):
        self.model.fit_transform(documents)

        return self.model

    def get_topics(self, n = 25):
        """
        n -- число лексем с наибольшим весом для выбора в каждой теме
        """
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

