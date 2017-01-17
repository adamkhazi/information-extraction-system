import tika
import nltk
import numpy
import untangle
import os

from tika import parser
from nltk.tokenize import word_tokenize

class TextExtractor:
    __dataset_raw_data_folder = "dataset_raw_data"
    __file_path_seperator = "/"

    __file_ext_txt = ".txt"
    __file_ext_xml = ".xml"
    __file_ext_pdf = ".pdf"
    __file_ext_doc = ".doc"

    # file names examples
    def populate_file_names(self):
        self.dataset_filenames = []
        for filename in os.listdir(self.__dataset_raw_data_folder):
            if filename.endswith(self.__file_ext_pdf) or filename.endswith(self.__file_ext_doc):
                self.dataset_filenames.append(os.path.splitext(filename)[0])

    def get_extracted_content(self):
        # files share an index
        file_content = []
        file_metadata = []

        for filename in enumerate(self.dataset_filenames):
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + self.__file_ext_txt
            extracted_information = parser.from_file(filepath)
            file_content.append(extracted_information["content"])
            file_metadata.append(extracted_information["metadata"])

        return file_content, file_metadata

    # xml[0].NewDataSet.Profile.cn_fname.cdata.strip()
    def read_xml_labelled_info(self):
        xml_labels = []
        for idx, filename in enumerate(self.dataset_filenames):
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename + self.__file_ext_xml
            xml_file = untangle.parse(filepath)
            xml_labels.append(xml_file)
        return xml_labels
"""
ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
"""
