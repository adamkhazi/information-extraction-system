import unittest
import pdb

import feature_generator
from feature_generator import FeatureGenerator
from dataset import Dataset
from we_model import WeModel

class FeatureGeneratorTests(unittest.TestCase):
    we_model = WeModel()
    w2v_model = we_model.load_pretrained_model() # optionally load a pretrained model here 

    def test_feature_doc_len(self):
        input, X, y = self.get_feature_generator_results()
        # document list
        input = [input]
        self.assertEqual(len(X), len(input))

    def test_feature_lines_len(self):
        input, X, y = self.get_feature_generator_results()
        # document list
        input = [input]
        self.assertEqual(len(X[0]), len(input[0]))

    def test_feature_tokens_same_len(self):
        input, X, y = self.get_feature_generator_results()
        # document list
        input = [input]

        input_token_nr = [1 for doc in input for line in doc for token in line]
        input_token_nr = sum(input_token_nr)

        X_token_nr = [1 for doc in X for line in doc for token in line]
        X_token_nr = sum(input_token_nr)

        self.assertEqual(input_token_nr, X_token_nr)

    def get_feature_generator_results(self):
        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")],
                [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]

        dataset = Dataset()
        word2count, word2idx = dataset.encode_dataset([input])

        f_generator = FeatureGenerator(self.w2v_model, word2count, word2idx)

        X = f_generator.generate_features_docs([input])
        y = f_generator.generate_true_outcome([input])

        # run tests on X and y
        return input, X, y
