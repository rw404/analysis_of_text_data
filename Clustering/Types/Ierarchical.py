from sklearn.cluster import AgglomerativeClustering

class HierarchicalClusters(object):

    def __init__(self):
        self.model = AgglomerativeClustering()

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        """
        Обучение агломеративной модели на данных
        """
        clusters = self.model.fit_predict(documents)
        self.labels = self.model.labels_
        self.children = self.model.children_

        return clusters

# Pipe
model = Pipeline([
    ('norm', TextNormalizer()),
    ('vect', OneHotVectorizer()),
    ('clusters', HierarchicalClusters())
])

model.fit_transform(docs)
labels = model.named_steps['clusters'].labels
pickles = list(corpus.fileids(categories = ['news']))

for idx, fileid in enumerate(pickles):
    print("Documnet '{}' assigned to cluster {}.".format(fileid, lables[idx]))

# Plots
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(children, **kwargs):
    # Расстояния между парами потомков
    distance = position = np.arange(children.shape[0])

    # Матрица связей
    linkage_matrix = np.column_stack([
        children, distance, position
    ]).astype(float)

    # Визуализация
    fig, ax = plt.subplots(figsize = (10, 5)) # set size
    ax = dendrogram(linkage_matrix, **kwargs)
    plt.tick_params(axis = 'x', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.tight_layout()
    plt.show()

children = model.named_steps['clusters'].children
plot_dendrogram(children)
