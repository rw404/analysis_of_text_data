import matplotlib
from nltk import sent_tokenize, word_tokenize

# Rentgen diagram
import json
import codecs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

with codecs.open('oz.json', 'r', 'utf-8-sig') as data:
    text = json.load(data)
    cast = text['cast']

    # График упоминаний персонажей в главах
    oz_words = []
    headings = []

    chap_lens = []
    for heading, chapter in text['chapters'].items():
        # Добаить название главы в список
        headings.append(heading)
        for sent in sent_tokenize(chapter):
            for word in word_tokenize(sent):
                # Добавить каждое слово
                oz_words.append(word)
        # Запомнить длину главы в словах
        chap_lens.append(len(oz_words))

    # Отметить начала глав
    chap_starts = [0] + chap_lens[:-1]
    # Объеденить с названием глав
    chap_marks = list(zip(chap_starts, headings))

    cast.reverse()
    points = []
    # Добавлять точку, если встертили персонажа
    for y in range(len(cast)):
        for x in range(len(oz_words)):
            # Для персонажей, у которых имена из одного слова
            if len(cast[y].split()) == 1:
                if cast[y] == oz_words[x]:
                    points.append((x, y))
            # Для имен из двух слов
            else:
                if cast[y] == ' '.join((oz_words[x-1], oz_words[x])):
                    points.append((x, y))
    if points:
        x, y = list(zip(*points))
    else:
        x = y = ()

    # Диаграмма
    fig, ax = plt.subplots(figsize = (12, 6))
    # Добавить вертикальные линии - маркеры окончания глав с подписями
    for chap in chap_marks:
        plt.axvline(x = chap[0], linestyle = '-',
                    color = 'gainsboro')
        plt.text(chap[0], -2, chap[1], size = 6, rotation = 90)
    # Точки упоминания персонажей
    plt.plot(x, y, "|", color = "darkorange", scalex = .1)
    plt.tick_params(
        axis = 'x', which = 'both', bottom = 'off', labelbottom = 'off'
    )
    plt.yticks(list(range(len(cast))), cast, size = 8)
    plt.ylim(-1, len(cast))
    plt.title("Character Mentions in th Wizard of Oz")
    plt.show()

    # Сохранение диаграммы
    plt.savefig('Rentgen_Diagram.png')
