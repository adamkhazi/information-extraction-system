import unittest

import tagger
from tagger import Tagger

#python3 -m unittest tests.tagger_tests
class TaggerTests(unittest.TestCase):
    """
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

"""
    def match_label_get_results(self, tagger):
        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")],
            [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]
        input_label = "Not a match"
        input_match_tag = "match"
        output = tagger.match_label(input, input_label, input_match_tag)
        return input, output

    def test_match_label_same_nr_lines(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if same nr of lines returned
        self.assertEqual(len(input), len(output))

    def test_match_label_same_nr_tokens(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if same nr of tokens returned
        input_nr_tuples = [len(line) for line in input]
        input_nr_tuples = sum(input_nr_tuples)

        output_nr_tuples = [len(line) for line in output]
        output_nr_tuples = sum(output_nr_tuples)
        self.assertEqual(input_nr_tuples, output_nr_tuples)

    def test_match_label_4_slots_each_tuple(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if all tuples contain 4 slots
        input_nr_tuple_slots = [1 for line in input for tuple in line if len(tuple) == 4]
        input_nr_tuple_slots = sum(input_nr_tuple_slots)

        output_nr_tuple_slots = [1 for line in output for tuple in line if len(tuple) == 4]
        output_nr_tuple_slots = sum(output_nr_tuple_slots)
        self.assertEqual(input_nr_tuple_slots, output_nr_tuple_slots)

    def test_match_label_same_token_strs_returned(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if same token strs returned in first slot of tuple 
        same_tokens_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[0] != input[line_idx][tuple_idx][0]:
                    same_tokens_returned = False

        self.assertEqual(same_tokens_returned, True)

    def test_match_label_pos_labels_not_altered(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if same token strs returned in first slot of tuple 
        same_pos_labels_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[1] != input[line_idx][tuple_idx][1]:
                    same_pos_labels_returned = False

        self.assertEqual(same_pos_labels_returned, True)

    def test_match_label_nonlocalner_labels_not_altered(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        # check if same token strs returned in first slot of tuple 
        same_nlner_labels_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[3] != input[line_idx][tuple_idx][3]:
                    same_nlner_labels_returned = False

        self.assertEqual(same_nlner_labels_returned, True)

