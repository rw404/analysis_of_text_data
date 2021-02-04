class BaseNgramModel(object):
    """
    BaseNgramModel создает модель языка n-грамм.
    """

    def __init__(self, ngram_counter):
        """
        BaseNgramModel инициализируется объектом NgramCounter
        """
        self.n = ngram_counter.n
        self.ngram_counter = ngram_counter
        self.ngrams = ngram_counter.ngrams[ngram_counter.n]
        self._check_against_vocab = self.ngram_counter._check_against_vocab

    def score(self, word, context):
        """
        Возвращает оценку максимальной вероятности, что данное слово продолжит этот контекст.

        fdist[context].freq(word) == fdist[(context, word)] / fdist[context]
        """
        context = self._check_against_vocab(context)

        return self.ngrams[context].freq(word)

    def check_context(self, context):
        """
        Проверяет длину контекста, которая должна быть меньше длины n-граммы высшего порядка модели.

        Контекст как кортеж.
        """
        if len(context) >= self.n:
            raise ValueError("Context too long for this n-gram")

        return tuple(context)

    def logscore(self, word, context):
        """
        Возвращает логарифм вероятности появления слова в данном контексте.
        """

        score = self.score(word, context)
        if score <= 0.0:
            return float("-inf")

        return log(score, 2)

    def entropy(self, text):
        """
        Перекрестная энтропия модели n-грамм для заданного текста в форме списка строк, разделенных
        запятыми.

        Средний логарифм вероятности всех слов в тексте.
        """
        normed_text = (self._check_against_vocab(word) for word in text)
        entropy = 0.0
        processed_ngrams = 0
        for ngram in self.ngram_counter.to_ngrams(normed_text):
            context, word = tuple(ngram[:-1]), ngram[-1]
            entropy += self.logscore(word, context)
            processed_ngrams += 1

        return -(entropy / processed_ngrams)

    def perplexity(self, text):
        """
        Неопределенность текста.
        """
        return pow(2.0, self.entropy(text))

trigram_model = BaseNgramModel(count_ngrams(3, vocab, sents))
fivegram_model = BaseNgramModel(count_ngrams(5, vocab, sents))

print(trigram_model.perplexity(sents[0]))
print(fivegram_model.perplexity(sents[0]))
