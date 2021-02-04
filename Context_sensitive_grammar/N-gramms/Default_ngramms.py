def ngrams(words, n = 2):
    for idx in range(len(words)-n+1):
        yield tuple(words[idx:idx+n])

words = [
    "The", "reporters", "listened", "closely", "as", "the", "President",
    "of", "the", "United", "States", "adressed", "the", "room", ".",
]

for ngram in ngrams(words, n = 3):
    print(ngram)

"""
('The', 'reporters', 'listened')
('reporters', 'listened', 'closely')
('listened', 'closely', 'as')
('closely', 'as', 'the')
('as', 'the', 'President')
('the', 'President', 'of')
('President', 'of', 'the')
('of', 'the', 'United')
('the', 'United', 'States')
('United', 'States', 'adressed')
('States', 'adressed', 'the')
('adressed', 'the', 'room')
('the', 'room', '.')
"""

# class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
#     ...
    def ngrams(self, n = 2, fileids = None, categories = None):
        for sent in self.sents(fileids = fileids, categories = categories):
            for ngram in nltk.ngrams(sent, n):
                yield ngram
#     ...

