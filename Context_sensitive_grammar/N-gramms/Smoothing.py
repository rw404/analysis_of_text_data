class AddKNgramModel(BaseNgramModel):
    """
    Обобщенное сглаживание Лапласа (add-k).
    """
    def __init__(self, k, *args):
        """
        k увеличивает счетчик слов в процессе оценки.
        """
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        context_count = context_freqdist.N()
        return (word_count + self.k)/(context_count + self.k_norm)

class LaplaceNgramModel(AddKNgramModel):
    """
    Сглаживание Лапласа.
    """
    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)

class KneserNeyModel(BaseNgramModel):
    """
    Сглаживание Кнесера - Нея
    """
    def __init__(self, *args):
        super(KneserNeyModel, self).__init(*args)
        self.model = nltk.KneserNeyProbDist(self.ngrams)

    def score(self, word, context):
        """
        KneserNeyProbDist из NLTK
        """
        trigram = tuple((context[0], context[1], word))
        return self.model.prob(trigram)

    def samples(self):
        return self.model.samples()

    def prob(self, sample):
        return self.model.prob(sample)

corpus = PickledCorpusReader('../corpus')
tokens = [''.join(word) for word in corpus.words()]
vocab = Counter(tokens)
sents = list([word[0] for word in sent] for sent in corpus.sents())

counter = count_ngrams(3, vocab, sents)
knm = KneserNeyModel(counter)

def complete(input_text):
    tokenized = nltk.word_tokenize(input_text)
    if len(tokenized) < 2:
        response = "Say more."
    else:
        completions = {}
        for sample in knm.samples():
            if (sample[0], sample[1]) == (tokenized[-2], tokenized[-1]):
                completions[sample[2]] = knm.prob(sample)
        if len(completions) == 0:
            response = "Can we takl about something else?"
        else:
            best = max(
                completions.keys(), key = (lambda key: completions[key])
            )
            tokenized += [best]
            response = " ".join(tokenized)

    return response

print(complete("The President of the United"))
print(complete("This election year will"))
