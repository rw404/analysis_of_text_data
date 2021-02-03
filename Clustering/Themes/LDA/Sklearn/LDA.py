from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import CountVectorizer

class SklearnTopicModels(object):

    def __init__(self, n_topics = 50):
        """
        n_topics - число тем
        """
        self.n_topics = n_topics
        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', CountVectorizer(tokenizer = identity,
                                     preprocessor = None, lowercase = False)),
            ('model', LatentDirichletAllocation(n_topics = n_topics)),
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

if __name__ == '__main__':
    corpus = PickledCorpusReader('corpus/')

    lda       = SklearnTopicModels()
    documents = corpus.docs()

    lda.fit_transform(documents)
    topics = lda.get_topics()
    for topic, terms in topics.items():
        print("Topic #{}:".format(topic+1))
        print(terms)

