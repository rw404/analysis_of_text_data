from nltk.chunk.regexp import RegexpParser


GRAMMAR = """
    S -> NNP VP
    VP -> V PP
    PP -> P NP
    NP -> DT N
    NNP -> 'Gwen' | 'George'
    V -> 'looks' | 'burns'
    P -> 'in' | 'for'
    DT -> 'the'
    N -> 'castle' | 'ocean'
    """

GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
chunker = RegexpParser(GRAMMAR)
