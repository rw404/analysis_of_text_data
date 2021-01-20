import codecs
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

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








