import tika
import nltk
import numpy
import untangle
import os

from tika import parser
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Extractor:
    __dataset_raw_data_folder = "dataset_raw_data"
    __file_path_seperator = "/"

    __file_ext_txt = ".txt"
    __file_ext_xml = ".xml"
    __file_ext_pdf = ".pdf"
    __file_ext_doc = ".doc"
    __file_ext_docx = ".docx"
    __file_ext_msg = ".msg"

    def get_job_titles(self, label_set):
        if not label_set.NewDataSet.Jobs:
            return []

        job_list = []
        Jobs = label_set.NewDataSet.Jobs
        for job in Jobs:
            if not job.job_position:
                continue
            jp = job.job_position.cdata.strip()
            job_list.append(jp)

        return job_list

    def get_dataset_folder(self):
        return self.__dataset_raw_data_folder

    # file names examples, nr_of_file: -1 is limitless
    def populate_file_names(self, nr_of_files=-1):
        self.dataset_filenames = []
        counter = 0
        for filename in os.listdir(self.__dataset_raw_data_folder):
            if filename.endswith(self.__file_ext_pdf) or filename.endswith(self.__file_ext_doc) or filename.endswith(self.__file_ext_docx) or filename.endswith(self.__file_ext_msg) :
                filename, file_ext = os.path.splitext(filename)
                self.dataset_filenames.append((filename, file_ext))
                counter += 1
                print(filename)
                if counter == nr_of_files and nr_of_files != -1:
                    break

    def read_resume_content(self):
        # files share an index
        file_content = []
        file_metadata = []

        for idx, filename in enumerate(self.dataset_filenames):
            # append filename + ext to path
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + filename[1]
            extracted_information = parser.from_file(filepath)
            file_content.append(extracted_information["content"])
            file_metadata.append(extracted_information["metadata"])

        self.resume_content = file_content
        self.resume_metadata = file_metadata

    # xml[0].NewDataSet.Profile.cn_fname.cdata.strip()
    def read_resume_labels(self):
        resume_labels = []
        for idx, filename in enumerate(self.dataset_filenames):
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + self.__file_ext_xml
            xml_file = untangle.parse(filepath)
            resume_labels.append(xml_file)
        self.resume_labels = resume_labels

    def tokenise_content_by_line(self):
        for idx, file_content in enumerate(self.resume_content):
            self.resume_content[idx] = file_content.splitlines()

    # tokenise and filter stop words
    def tokenise_content_by_words(self):
        rtokenizer = RegexpTokenizer(r'\w+')
        for doc_idx, doc in enumerate(self.resume_content):
            tokenized_doc_lines = []
            for line_idx, line in enumerate(doc):
                line = rtokenizer.tokenize(line)
                if line != []:
                    filtered_words = [word for word in line if word not in stopwords.words('english')]
                    tokenized_doc_lines.append(filtered_words)
            self.resume_content[doc_idx] = tokenized_doc_lines

    def remove_empty_resumes(self):
        for idx, file_content in enumerate(self.resume_content):
            if file_content is None:
                del self.dataset_filenames[idx]
                del self.resume_content[idx]

    def prepare_dataset(self):
        self.populate_file_names(nr_of_files=1500)
        self.read_resume_content()
        self.remove_empty_resumes()
        self.read_resume_labels()
        self.tokenise_content_by_line()
        self.tokenise_content_by_words()

"""
ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
"""
