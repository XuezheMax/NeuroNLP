__author__ = 'max'


class Instance(object):
    def __init__(self, words, word_ids, postags, pos_ids, heads, types, type_ids):
        self.words = words
        self.word_ids = word_ids
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return len(self.words)
