import unittest

import tokeniser
from tokeniser import Tokeniser

#python3 -m unittest tests.tokeniser_tests
class TokeniserTests(unittest.TestCase):
    def test_tokenise_lines(self):
        tokeniser = Tokeniser()

        # each slot is résumé plain text
        input_docs = ["sample resume output\rsample resume output",
                "\rsample resume output\rsample resume output",
                "sample resume output\nsample resume output",
                "\nsample resume output\nsample resume output"]

        # each slot has a list of lines found in each résumé inputted
        correct_output = [["sample resume output", "sample resume output"],
                ["", "sample resume output", "sample resume output"],
                ["sample resume output", "sample resume output"],
                ["", "sample resume output", "sample resume output"]]

        output = tokeniser.tokenise_docs_to_lines(input_docs)

        self.assertEqual(output, correct_output)

    def test_tokenise_words(self):
        tokeniser = Tokeniser()

        # each slot is a line within a résumé 
        input_lines = [["sample resume output sample resume output",
                "  sample resume output sample resume output  ",
                "sample resume output.            sample resume output",
                ""]]

        # each slot is a token 
        correct_output = [[["sample", "resume", "output", "sample", "resume", "output"],
                ["sample", "resume", "output", "sample", "resume", "output"],
                ["sample", "resume", "output", "sample", "resume", "output"]
                ]]

        output = tokeniser.tokenise_doclines_to_words(input_lines)

        self.assertEqual(output, correct_output)

