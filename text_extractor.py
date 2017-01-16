import tika
import nltk
import numpy

from tika import parser
from nltk.tokenize import word_tokenize

class TextExtraction:
    self.__dataset_raw_data_folder = "dataset_raw_data"
    self.__file_path_seperator = "/"

    def get_dataset(self):
        dataset_raw_files = []
        for filename in os.listdir(self.__dataset_raw_data_folder):
            current_file_path = self.__dataset_raw_data_folder + self.__file_path_seperator + filename
            dataset_raw_data.append(current_file_path)

        # files share an index
        file_content = []
        file_metadata = []

        for file in dataset_raw_data:
            extracted_information = parser.from_file(file)
            file_content.append(extracted_information["content"])
            file_metadata.append(extracted_information["metadata"])

        return file_content, file_metadata

"""
ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
"""
