import unittest

import tagger
from tagger import Tagger

#python3 -m unittest tests.tagger_tests
class TaggerTests(unittest.TestCase):

    ### POS ###
    def pos_tag_get_results(self, tagger):
        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""),
            ("test", "", "", "")], [("this", "", "", ""), ("is", "", "", ""),
                ("a", "", "", ""), ("test", "", "", "")]]
        output = tagger.pos_tag(input)
        return input, output

    # check if same nr of lines returned
    def test_pos_tag_nr_lines(self):
        tagger = Tagger()
        input, output = self.pos_tag_get_results(tagger)
        self.assertEqual(len(input), len(output))

    ### MATCH LABELS - WINDOW MATCHING  ###
    def match_label_get_results(self, tagger):
        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""),
            ("test", "", "", "")], [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]
        input_label = "Not a match"
        input_match_tag = "match"
        output = tagger.match_label(input, input_label, input_match_tag)
        return input, output

    # check if same nr of lines returned
    def test_match_label_same_nr_lines(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        self.assertEqual(len(input), len(output))

    # check if same nr of tokens returned
    def test_match_label_same_nr_tokens(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        input_nr_tuples = [len(line) for line in input]
        input_nr_tuples = sum(input_nr_tuples)

        output_nr_tuples = [len(line) for line in output]
        output_nr_tuples = sum(output_nr_tuples)
        self.assertEqual(input_nr_tuples, output_nr_tuples)

    # check if all tuples contain 4 slots
    def test_match_label_4_slots_each_tuple(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        input_nr_tuple_slots = [1 for line in input for tuple in line if len(tuple) == 4]
        input_nr_tuple_slots = sum(input_nr_tuple_slots)

        output_nr_tuple_slots = [1 for line in output for tuple in line if len(tuple) == 4]
        output_nr_tuple_slots = sum(output_nr_tuple_slots)
        self.assertEqual(input_nr_tuple_slots, output_nr_tuple_slots)

    # check if same token strs returned in first slot of tuple 
    def test_match_label_same_token_strs_returned(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        same_tokens_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[0] != input[line_idx][tuple_idx][0]:
                    same_tokens_returned = False

        self.assertEqual(same_tokens_returned, True)

    # check if same token strs returned in first slot of tuple 
    def test_match_label_pos_labels_not_altered(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        same_pos_labels_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[1] != input[line_idx][tuple_idx][1]:
                    same_pos_labels_returned = False

        self.assertEqual(same_pos_labels_returned, True)

    # check if same token strs returned in first slot of tuple 
    def test_match_label_nonlocalner_labels_not_altered(self):
        tagger = Tagger()
        input, output = self.match_label_get_results(tagger)

        same_nlner_labels_returned = True
        for line_idx, line in enumerate(output):
            for tuple_idx, tuple in enumerate(line):
                if tuple[3] != input[line_idx][tuple_idx][3]:
                    same_nlner_labels_returned = False

        self.assertEqual(same_nlner_labels_returned, True)

    # check IOB was applied correctly for entities found
    def test_match_label_IOB_applied_correctly(self):
        tagger = Tagger()

        input =  [[("Brunel", "", "", ""), ("University", "", "", ""), ("test", "", "", ""), ("test", "", "", "")], [("test", "", "", ""), ("test", "", "", ""), ("Brunel", "", "", ""), ("University", "", "", "")], [("test", "", "", ""), ("test", "", "", ""), ("Brunel", "", "", ""), ("University", "", "", "")]]

        input_label = "Brunel University"
        input_match_tag = "match"
        output = tagger.match_label(input, input_label, input_match_tag)
        output = tagger.match_label(output, input_label, input_match_tag)
        output = tagger.match_label(output, input_label, input_match_tag)
        output = tagger.add_default_entity_tags(output)

        correct_iob = True
        for line in output:
            for token_idx, token in enumerate(line):
                if token[3].split("-", 1)[0] == "O":
                    next_token = "EOL" if len(line) == token_idx+1 else line[token_idx+1][3].split("-", 1)[0]
                    if next_token == "I":
                        correct_iob = False
        self.assertEqual(correct_iob, True)

"""
    ### NONLOCAL NER ###
    def test_nonlocal_ner_tag(self):
        tagger = Tagger()

        input =  [[("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")],
            [("this", "", "", ""), ("is", "", "", ""), ("a", "", "", ""), ("test", "", "", "")]]

        correct_output = [[("this", "O", "", ""), ("is", "O", "", ""), ("a", "O", "", ""), ("test", "O", "", "")],
                [("this", "O", "", ""), ("is", "O", "", ""), ("a", "O", "", ""), ("test", "O", "", "")]]

        output = tagger.nonlocal_ner_tag(input)
"""
