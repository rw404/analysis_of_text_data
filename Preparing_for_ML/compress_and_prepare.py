import pickle
import os
import time
import bs4
import logging
import codecs
from readability.readability import Unparseable
from readability.readability import Document as Paper
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag
import nltk

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']

class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    Корпус чтения HTML-документов для доп. предварительной обработки 
    """

    def __init__(self, root, fileids = DOC_PATTERN, encoding = 'utf8',
            tags = TAGS, **kwargs):
        """
        Инициализация объектов чтения корпуса.
        Аргументы, управляющие классификацией
        ("cat_pattern", "cat_map", "cat_file"), передаются в конструктор CategorizedCorpusReader.
        Остальные аргументы передаются в CorpusReader.
        """

        #Если не передан в класс явно, то добавляется шаблон категорий
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        #Инициализация корпусов - предков
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        #Сохранить тэги
        self.tags = tags

    def resolve(self, fileids, categories):
        """
        Возвращает список идентификаторов файлов или названий категорий, которые передаются каждой
        внутренней функции объекта чтения корпуса
        """

        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids = None, categories = None):
        """
        Полный текст HTML-документа, закрывая его после прочтения
        """

        #Список файлов
        fileids = self.resolve(fileids, categories)

        #Генератор, загружающий документы в память по одному
        for path, encoding in self.abspath(fileids, include_encoding = True):
            with codecs.open(path, "r", encoding = encoding) as f:
                yield f.read()

    def sizes(self, fileids = None, categories = None):
        """
        Список кортежей, идентификаторов файла и его размер. -- мониторинг корпуса
        """

        fileids = self.resolve(fileids, categories)

        for path in self.abspath(fileids):
            yield path, os.path.getsize(path)

    def html(self, fileids = None, categories = None):
        """
        Содержимое HTML каждого документа, очищая его с помощью библиотеки readability-lxml
        """
        
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue

    def paras(self, fileids = None, categories = None):
        """
        Использует BeautifulSoup для выделения абзацев из HTML
        """

        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(tags):
                yield element.text
            soup.decompose()
    
    def sents(self, fileids = None, categories = None):
        """
        Механизм выделения предложений и абзацев
        """

        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids = None, categories = None):
        """
        Выделение слов из предложений
        """

        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids = None, categories = None):
        """
        Сегментизирует, лексемизирует и маркирует документ
        """

        for paragraph in self.paras(fileids = fileids):
            yield [
                    pos_tag(wordpunct_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                    ]

    def describe(self, fileids = None, categories = None):
        """
        Оценка словаря
        """
        started = time.time()

        # Структура для подсчета.
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()
        
        # Обход абзацев, лексемы и подсчет
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in para:
                counts['sents'] += 1
                for word, tag in sent:
                    counts['words'] += 1
                    tokens[word] += 1

        # Определить число файлов и категорий в корпусе
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Вернуть структуру
        return {
                'files': n_fileids,
                'topics': n_topics,
                'paras': counts['paras'],
                'sents': counts['sents'],
                'words': counts['words'],
                'vocab': len(tokens),
                'lexdiv': float(len(tokens)) / float(counts['words']),
                'ppdoc': float(counts['paras']) / float(n_fileids),
                'sppar': float(counts['sents']) / float(counts['paras']),
                'secs': time.time() - started,
                }

class Preprocessor(object):
    """
    Обертка 'HTMLCorpusReader' и выполнить лексемизацию 
    с маркировкой частями речи.
    """
    
    def __init__(self, corpus, target = None, **kwargs):
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids = None, categories = None):
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        # Найти путь к каталогу относительно корня исходного корпуса.
        parent = os.path.relpath(
                os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
                )

        # Выделить части пути для реконструирования
        basename = os.path.basename(fileid)
        # Имя.формат
        name, ext = os.path.splitext(basename)

        # Сконструировать имя файла с расширением .pickle
        basename = name + '.pickle'

        # Вернуть путь к файлу относительно корня целевого корпуса.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        for paragraph in self.corpus.paras(fileids = fileid):
            yield [
                    pos_tag(wordpunct_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                    ]

    def process(self, fileid):
        """
        Проверка документа и сохранение докумнта на диске
        """

        # Определить путь к файлу для записи результата.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Существование каталога
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Убедиться, что parent - каталог
        if not os.path.isdir(parent):
            raise ValueError(
                    "Please supply a directory to write preprocessed data to."
                    )

        # Создать структуру данных для записи в архив
        with open(target, 'wb') as f:
            pickle.dump(documnet, f, pickle.HIGHEST_PROTOCOL)

        # Удалить документ из памяти
        del document

        # Вернуть путь к целевому файлу
        return target

    def transform(self, fileids = None, categories = None):
        # Создать целовой каталог, если его нет
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Получить имена файлов для обработки
        for fileid in self.fileids(fileids, categories):
            yield self.process(fileid)


class PickledCorpusReader(HTMLCorpusReader):

    def __init__(delf, root, fileids = PKL_PATTERN, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def docs(self, fileids = None, categories = None):
        fileids = self.resolve(fileids, categories)
        # Загружатьь документы по одному.
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids = None, categories = None):
        for doc in self.docs(fileids, categories):
            for para in doc:
                yield para

    def sents(self, fileids = None, categories = None):
        for para in self.paras(fileids, categories):
            for sent in para:
                yield sent

    def tagged(self, fileids = None, categories = None):
        for sent in self.sents(fileids, categories):
            for tagged_token in sent:
                yield tagged_token

    def words(self, fileids = None, categories = None):
        for tagged in self.tagged(fileids, categories):
            yield tagged[0]


