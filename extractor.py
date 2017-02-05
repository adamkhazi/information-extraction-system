import tika
import nltk
import numpy
import os
import html

import xml.etree.cElementTree as ET

from tika import parser
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from logger import Logger

# This class extracts content from resume files of various formats.
# Also it extracts content from XML files and turns them into Python objects.
class Extractor:
    __dataset_raw_data_folder = "dataset_raw_data"
    __file_path_seperator = "/"

    __file_ext_txt = ".txt"
    __file_ext_xml = ".xml"
    __file_ext_pdf = ".pdf"
    __file_ext_doc = ".doc"
    __file_ext_docx = ".docx"
    __file_ext_msg = ".msg"

    def __init__(self):
        self.logger = Logger()
        self.logger.println("extractor created")

    def get_edu_majors(self, label_set):
        edu_major_list = label_set.findall("Education/edu_major")
        if not edu_major_list:
            return []
        else:
            return [html.unescape(major.text) for major in edu_major_list if major.text is not None]

    def get_edu_institutions(self, label_set):
        edu_inst_list = label_set.findall("Education/edu_inst_name")
        if not edu_inst_list:
            return []
        else:
            return [html.unescape(inst.text) for inst in edu_inst_list if inst.text is not None]

    def get_company_names(self, label_set):
        company_list = label_set.findall("Jobs/job_company_name")
        if not company_list:
            return []
        else:
            return [html.unescape(company.text) for company in company_list if company.text is not None]

    def get_job_titles(self, label_set):
        job_list = label_set.findall("Jobs/job_position")
        if not job_list:
            return []
        else:
            return [html.unescape(j_title.text) for j_title in job_list if j_title.text is not None]

    def get_dataset_folder(self):
        return self.__dataset_raw_data_folder

    # file names examples, nr_of_file: -1 is limitless
    def __populate_file_names(self, nr_of_files=-1):
        self.dataset_filenames = []
        counter = 0
        for filename in os.listdir(self.__dataset_raw_data_folder):
            if filename.endswith(self.__file_ext_pdf) or filename.endswith(self.__file_ext_doc) or filename.endswith(self.__file_ext_docx):
                filename, file_ext = os.path.splitext(filename)
                self.dataset_filenames.append((filename, file_ext))
                counter += 1
                if counter == nr_of_files and nr_of_files != -1:
                    break
        self.logger.println("read %s file names" % counter)

    def __read_resume_content(self):
        self.resume_content = []
        self.resume_metadata = []
        # idxs of files that don't have content
        remove_files_idxs = []
        for idx, filename in enumerate(self.dataset_filenames):
            self.logger.println("sending resume %s/%s to tika" % (idx, len(self.dataset_filenames)-1) )
            # append filename + ext to path
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + filename[1]
            extracted_information = parser.from_file(filepath)

            # check if a supported file was processed successfully
            if "content" in extracted_information:
                self.resume_content.append(extracted_information["content"])
                self.resume_metadata.append(extracted_information["metadata"])
            else:
                remove_files_idxs.append(idx)

        for idx in remove_files_idxs:
            self.logger.println("removing unprocessed resume file at index %s named %s" % (idx, self.dataset_filenames[idx]))
            del self.dataset_filenames[idx]

        self.logger.println("read content from %s resume files" % len(self.resume_content))

    def __read_resume_labels(self):
        resume_labels = []
        for idx, filename in enumerate(self.dataset_filenames):
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + self.__file_ext_xml
            xml_file = ET.ElementTree(file=filepath)
            resume_labels.append(xml_file)
        self.resume_labels = resume_labels
        self.logger.println("read labels from %s xml files" % len(self.resume_labels))

    def __remove_empty_resumes(self):
        # idxs of files that don't have content
        remove_files_idxs = []
        for idx, file_content in enumerate(self.resume_content):
            if file_content is None:
                remove_files_idxs.append(idx)

        deleted_count = 0
        for idx in remove_files_idxs:
            self.logger.println("removing empty resume file at index %s named %s" % (idx, self.dataset_filenames[idx]))
            del self.dataset_filenames[idx-deleted_count]
            del self.resume_metadata[idx-deleted_count]
            del self.resume_content[idx-deleted_count]
            deleted_count += 1
        self.logger.println("removed empty resume files and total file count is at %s" % len(self.resume_content))

    def read_raw_files(self, nr_of_docs):
        self.__populate_file_names(nr_of_docs)
        self.__read_resume_content()
        self.__remove_empty_resumes()
        self.__read_resume_labels()
        return self.resume_content, self.resume_labels

    # method is required when parsing and tagging a single file, mostly for
    # demonstration
    def read_resume_content(self, filepath):
        extracted_information = parser.from_file(filepath)
        docs = []
        docs.append(extracted_information["content"])
        return docs

"""
ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
"""
