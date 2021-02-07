import networkx as nx
from nltk.corpus import wordnet as wn

def graph_synsets(terms, pos = wn.NOUN, deph = 2):
    """
    Создает из терминов terms граф networkx с глубиной deph
    """
    G = nx.Graph(
        name = "WordNet Synsets Graph for {}".format(", ".join(terms)),
        deph = deph,
    )

    def add_term_links(G, term, current_depth):
        for syn in wn.synsets(term):
            for name in syn.lemma_names():
                G.add_edge(term, name)
                if current_depth < deph:
                    add_term_links(G, name, current_depth+1)
    for term in terms:
        add_term_links(G, term, 0)

    return G

G = graph_synsets(["trinket"])
print(nx.info(G))

import matplotlib.pyplot as plt

def draw_text_graph(G):
    pos = nx.spring_layout(G, scale = 18)
    nx.draw_networkx_nodes(
        G, pos, node_color = "white", linewidths = 0, node_size = 500
    )
    nx.draw_networkx_labels(G, pos, font_size = 10)
    nx.draw_networkx_edges(G, pos, edge_color = 'lightgrey')

    plt.tick_params(
        axis = 'both',      # изменяются обе оси, X и Y
        which = 'both',     # затрагиваются большие и маые деления
        bottom = 'off',     # скрыть деления вдоль нижнего края
        left = 'off',       # скрыть деления вдоль левого края
        labelbottom = 'off',# скрыть подписи вдоль нижнего края
        labelleft = 'off'   # скрыть подписи вдоль левого края
    )

    plt.show()
    plt.savefig('graph.png')

draw_text_graph(graph_synsets(['trinket']))

# EntityPairs
import itertools
from sklearn.base import BaseEstimator, TransformerMixin

class EntityPairs(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(EntityPirs, self).__init__()

    def pairs(self, document):
        return list(itertools.permutations(sent(document), 2))

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        return [self.pairs(document) for document in documents]

# GraphExtractor
import networkx as nx

class GraphExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.G = nx.Graph()

    def fit(self, documents, labels = None):
        return self

    def transform(self, documents):
        for document in documents:
            for first, second in document:
                if (first, second) in self.G.edges():
                    self.G.edges[(first, second)]['weights'] += 1
                else:
                    self.G.add_edge(first, second, weight = 1)
    return self.G

if __name__ == '__main__':
    from reader import PickledCorpusReader
    from sklearn.pipeline import Pipeline

    corpus = PickledCorpusReader('../corpus')
    docs = corpus.docs()

    graph = Pipeline([
        ('entities', EntityExtractor()),
        ('pairs', EntityPairs()),
        ('graph', GraphExtractor())
    ])

    G = graph.fit_transform(docs)
    print(nx.info(G))

# Centres
import heapq
from operator import itemgetter

def nbest_centrality(G, metrics, n = 10):
    # Вычисляет оценку центральности для каждой вершины
    nbest = {}
    for name, metric in metrics.items():
        scores = metric(G)

        # Запись оценки в свойство узла
        nx.set_node_attributes(G, name = name, values = scores)

        # Найти n топовых узлов
        topn = heapq.nlargest(n, scores.items(), key = itemgetter(1))
        nbest[name] = topn

    return nbest

# Разные метрики
from tabulate import tabulate

corpus = PickledCorpusReader('../corpus')
docs = corpus.docs()

graph = Pipeline([
    ('entities', EntityExtractor()),
    ('pairs', EntityPairs()),
    ('graph', GraphExtractor())
])

G = graph.fit_transform(docs)

centralities = {"Degree Centrality" : nx.degree_centrality,
                "Betweenness Centrality" : nx.betweenness_centrality}

centrality = nbest_centrality(G, centralities, 10)

for measure, scores in centrality.items():
    print("Ranking for {}".format(measure))
    print((tabulate(scores, headers = ["Top Terms", "Score"])))
    print("")

# Structure analysis
H = nx.ego_graph(G, "hollywood")
edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())
pos = nx.spring_layout(H, k = 0.3, iterations = 40)

nx.draw(
    H, pos, node_color = "skyblue", node_size = 20, edgelist = edges,
    edge_color = weights, width = 0.25, edge_cmap = plt.cm.Pastel2,
    with_labels = True, font_size = 6, alpha = 0.8
)
plt.show()

# Distributions
import seaborn as sns

sns.displot([G.degree(v) for v in G.nodes()], norm_hist = True)
plt.show()

# EntitySolver
import networkx as nx
from itertools import combinations

def pairwise_comparions(G):
    """
    Создает генератор пар узлов.
    """
    return combinations(G.nodes(), 2)

# Blocking
def edge_blocked_comparisons(G):
    """
    Генератор попарных сравнений, который выявляет вероятно подобные узлы, связанные ребрами с одной
    и той же сущностью.
    """
    for n1, n2 in pairwise_comparions(G):
        hood1 = frozenset(G.neighbors(n1))
        hood2 = frozenset(G.neighbors(n2))
        if hood1 & hood2:
            yield n1, n2

# Similarity
from fuzzywuzzy import fuzz

def similarity(n1, n2):
    """
    Возвращает среднюю оценку подобия
    """
    scores = [
        fuzz.partial_ratio(n1, n2),
        fuzz.partial_ratio(G.node[n1]['type'], G.node[n2]['type'])
    ]

    return float(sum(s for s in scores)) / float(len(scores))

def fuzzy_blocked_comprassions(G, threshold = 65):
    """
    Генератор попарных сравнений, выявляющий подобные узлы
    """
    for n1, n2 in pairwise_comparions(G):
        hood1 = frozenset(G.neighbors(n1))
        hood2 = frozenset(G.neighbors(n2))
        if hood1 & hood2:
            if similarity(n1, n2) > threshold:
                yield n1, n2

def info(G):
    """
    Обертка для nx.info с несколькими вспомогательными функциями.
    """
    pairwise = len(list(pairwise_comparions(G)))
    edge_blocked = len(list(edge_blocked_comparisons(G)))
    fuzz_blocked = len(list(fuzzy_blocked_comprassions(G)))

    output = [""]
    output.append("Number of Pairwise Comparisons: {}".format(pairwise))
    output.append("Number of Edge Blocked Comparisons: {}".format(edge_blocked))
    output.append("Number of Fuzzey Blocked Comparisons: {}".format(fuzz_blocked))

    return nx.info(G) + '\n'.join(output)

from sklearn.base import BaseEstimator, TransformerMixin

class FuzzyBlocker(BaseEstimator, TransformerMixin):

    def __init__(self, threshold = 65):
        self.threshold = threshold

    def fit(self, G, y = None):
        return self

    def transform(self, G):
        return fuzzy_blocked_comprassions(G, self.threshold)

