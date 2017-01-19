import unittest
import glob

import extractor
from extractor import Extractor

#python3 -m unittest tests.extractor_tests
class ExtractorTests(unittest.TestCase):

    def test_read_resume_content(self):
        extractor = Extractor()
        extractor.populate_file_names()
        extractor.read_resume_content()

        # content is of right type and not empty
        for document in extractor.resume_content:
            self.assertIsInstance(document, str)
            self.assertTrue(len(document) > 0)

        # correct number of file extracted
        directory = extractor.get_dataset_folder()

        number_of_files = len(glob.glob1(directory,"*.doc"))
        number_of_files += len(glob.glob1(directory,"*.docx"))
        number_of_files += len(glob.glob1(directory,"*.pdf"))

        number_of_extracted_files = len(extractor.resume_content)
        self.assertTrue(number_of_files == number_of_extracted_files)

