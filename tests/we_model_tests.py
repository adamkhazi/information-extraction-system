import unittest
import spacy

from we_model import WeModel

class WeModelTests(unittest.TestCase):
    we_model = WeModel()
    nlp = we_model.load_spacy()

    def test_load_pretrained_model(self):
        self.assertTrue(type(self.nlp) == spacy.en.English)

    def test_embedding_dimen(self):
        test_word = "word"
        word_vector = self.we_model.get_spacy_vec(self.nlp, test_word)
        self.assertEqual(len(word_vector), 300)
