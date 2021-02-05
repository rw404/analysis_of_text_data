import itertools
from nltk import sent_tokenize

def cooccurrence(text, cast):
    """
    На входе словарь text с главами {название: текст}
    и cast - список имен персонажей, разденных запятыми.
    Возвращает словарь счетчиков совхождений для всех возможных пар имен.
    """
    possible_pairs = list(itertools.combinations(cast, 2))
    cooccurring = dict.fromkeys(possible_pairs, 0)
    for title, chapter in text['chapters'].items():
        for sent in sent_tokenize(chapter):
            for pair in possible_pairs:
                if pair[0] in sent and pair[1] in sent:
                    cooccurring[pair] += 1
    return cooccurring

# Network Diagram
import json
import codecs
import networkx as nx
import matplotlib.pyplot as plt

with codecs.open('oz.json', 'r', 'utf-8-sig') as data:
    text = json.load(data)
    cast = text['cast']

    G = nx.Graph()
    G.name = "The Social Network of Oz"

    pairs = cooccurrence(text, cast)
    for pair, wgt in pairs.items():
        if wgt > 0:
            G.add_edge(pair[0], pair[1], weight = wgt)

    # Поместим Dorothy в центр
    D = nx.ego_graph(G, "Dorothy")
    edges, weights = zip(*nx.get_edge_attributes(D, "weight").items())

    # Добавление узлов
    pos = nx.spring_layout(D, k = 5, iterations = 40)
    nx.draw(D, pos, node_color = "gold", node_size = 50, edgelist = edges,
            width = .5, edge_color = "orange", with_labels = True, font_size = 12)
    plt.show()

    # Сохранение в файл
    plt.savefig('Network_Diagram_OZ.png')

