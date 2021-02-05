import matplotlib
from nltk import sent_tokenize

def matrix(text, cast):
    mtx = []
    for first in cast:
        row = []
        for second in cast:
            count = 0
            for title, chapter in text['chapters'].items():
                for sent in sent_tokenize(chapter):
                    if first in sent and second in sent:
                        count += 1
            row.append(count)
        mtx.append(row)
    return mtx

# Network Diagram
import json
import codecs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

with codecs.open('oz.json', 'r', 'utf-8-sig') as data:
    text = json.load(data)
    cast = text['cast']

    # Создаем матрицу с сортировкой по частоте
    mtx = matrix(text, cast)

    # Создание диаграммы
    fig, ax = plt.subplots()
    fig.suptitle('Character Co-occurence in Wizard of Oz', fontsize = 12)
    fig.subplots_adjust(wspace = .75)

    n = len(cast)
    x_tick_marks = np.arange(n)
    y_tick_marks = np.arange(n)

    ax1 = plt.subplot(121)
    ax1.set_xticks(x_tick_marks)
    ax1.set_yticks(y_tick_marks)
    ax1.set_xticklabels(cast, fontsize = 8, rotation = 90)
    ax1.set_yticklabels(cast, fontsize = 8)
    ax1.xaxis.tick_top()
    ax1.set_xlabel("By Frequency")
    plt.imshow(mtx,
               norm = matplotlib.colors.LogNorm(),
               interpolation = 'nearest',
               cmap = 'YlOrBr')

    # По алфавиту
    alpha_cast = sorted(cast)
    alpha_mtx = matrix(text, alpha_cast)

    ax2 = plt.subplot(122)
    ax2.set_xticks(x_tick_marks)
    ax2.set_yticks(y_tick_marks)
    ax2.set_xticklabels(alpha_cast, fontsize = 8, rotation = 90)
    ax2.set_yticklabels(alpha_cast, fontsize = 8)
    ax2.xaxis.tick_top()
    ax2.set_xlabel("By Frequency")
    plt.imshow(alpha_mtx,
               norm = matplotlib.colors.LogNorm(),
               interpolation = 'nearest',
               cmap = 'YlOrBr')

    plt.show()
    plt.savefig('Matrix.png')
