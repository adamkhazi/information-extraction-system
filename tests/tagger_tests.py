import unittest

import tagger
from tagger import Tagger

#python3 -m unittest tests.tokeniser_tests
class TaggerTests(unittest.TestCase):
    def test_pos_tag(self):
        tagger = Tagger()

        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")],
            [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]

        correct_output = [[("this", "DT", "", ""), ("is", "VBZ", "", ""), ("a", "DT", "", ""), ("test", "NN", "", "")],
            [("this", "DT", "", ""), ("is", "VBZ", "", ""), ("a", "DT", "", ""), ("test", "NN", "", "")]]

        output = tagger.pos_tag(input)

        self.assertEqual(output, correct_output)

    def test_nonlocal_ner_tag(self):
        tagger = Tagger()

        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")],
            [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]

        correct_output = [[("this", "O", "", ""), ("is", "O", "", ""), ("a", "O", "", ""), ("test", "O", "", "")],
                [("this", "O", "", ""), ("is", "O", "", ""), ("a", "O", "", ""), ("test", "O", "", "")]]

        output = tagger.nonlocal_ner_tag(input)
        self.assertEqual(output, correct_output)