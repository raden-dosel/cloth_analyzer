from nlpaug import Augmenter

class SynonymAugmenter:
    def __init__(self):
        self.aug = Augmenter('word2vec', model_path='glove.6B.100d')
    
    def augment(self, text):
        return self.aug.augment(text)