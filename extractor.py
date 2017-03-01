import tika
import nltk
import numpy
import os
import html
import textract
import pdb

import xml.etree.cElementTree as ET
from os.path import expanduser

from tika import parser
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from logger import Logger
from tags import Tags
home = expanduser("~") + "/"
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ['CLASSPATH'] = home + "tika-app-1.14.jar"
#from jnius import autoclass

# This class extracts content from resume files of various formats.
# Also it extracts content from XML files and turns them into Python objects.
class Extractor(Tags):
    __dataset_raw_data_folder = "dataset_raw_data"
    __file_path_seperator = "/"
    __empty_str = ''

    __file_ext_txt = ".txt"
    __file_ext_xml = ".xml"
    __file_ext_pdf = ".pdf"
    __file_ext_doc = ".doc"
    __file_ext_docx = ".docx"
    __file_ext_msg = ".msg"

    def __init__(self):
        self.logger = Logger()
        self.logger.println("extractor created")

    def get_edu_institutions_from_list(self, tuple_list):
        filtered = []
        for cur_tuple in tuple_list:
            if cur_tuple[1].split('-', 1)[::-1][0] == self._Tags__education_institution_tag:
                filtered.append(cur_tuple[0])
        return filtered

    def get_edu_major_from_list(self, tuple_list):
        filtered = []
        for cur_tuple in tuple_list:
            if cur_tuple[1].split('-', 1)[::-1][0] == self._Tags__education_course_tag:
                filtered.append(cur_tuple[0])
        return filtered

    def get_company_names_from_list(self, tuple_list):
        filtered = []
        for cur_tuple in tuple_list:
            if cur_tuple[1].split('-', 1)[::-1][0] == self._Tags__job_company_tag:
                filtered.append(cur_tuple[0])
        return filtered

    def get_company_position_from_list(self, tuple_list):
        filtered = []
        for cur_tuple in tuple_list:
            if cur_tuple[1].split('-', 1)[::-1][0] == self._Tags__job_position_tag:
                filtered.append(cur_tuple[0])
        return filtered

    def replace_dash(self, label):
        if label.rstrip().lstrip() == "-":
            return ""
        else:
            return label

    def get_edu_institutions_zy(self, label_set):
        edu_inst_list = label_set.findall(".//{http://tempuri.org/}SegrigatedQualification/{http://tempuri.org/}EducationSplit/{http://tempuri.org/}University")
        edu_inst_list = [inst.text for inst in edu_inst_list]
        return [inst for inst in edu_inst_list if inst is not self.__empty_str and inst is not None]

    def get_edu_majors_zy(self, label_set):
        edu_majors_list = label_set.findall(".//{http://tempuri.org/}SegrigatedQualification/{http://tempuri.org/}EducationSplit/{http://tempuri.org/}Degree")
        edu_majors_list = [major.text for major in edu_majors_list]
        return [major for major in edu_majors_list if major is not self.__empty_str and major is not None]

    def get_company_names_zy(self, label_set):
        company_list = label_set.findall(".//{http://tempuri.org/}SegrigatedExperience/{http://tempuri.org/}WorkHistory/{http://tempuri.org/}Employer")
        company_list = [company.text for company in company_list]
        return [company for company in company_list if company is not self.__empty_str and company is not None]

    def get_job_titles_zy(self, label_set):
        jtitle_list = label_set.findall(".//{http://tempuri.org/}SegrigatedExperience/{http://tempuri.org/}WorkHistory/{http://tempuri.org/}JobProfile")
        jtitle_list = [jtitle.text for jtitle in jtitle_list]
        return [jtitle for jtitle in jtitle_list if jtitle is not self.__empty_str and jtitle is not None]

    def get_edu_majors(self, label_set):
        edu_major_list = label_set.findall("Education/edu_major")
        edu_major_list = [edu_major.text for edu_major in edu_major_list]
        edu_major_list = [self.replace_dash(edu_major) for edu_major in edu_major_list if edu_major is not None]
        if not edu_major_list:
            return []
        else:
            return [html.unescape(major) for major in edu_major_list if major is not self.__empty_str and major is not None]

    def get_edu_institutions(self, label_set):
        edu_inst_list = label_set.findall("Education/edu_inst_name")
        edu_inst_list = [edu_inst.text for edu_inst in edu_inst_list]
        edu_inst_list = [self.replace_dash(edu_inst) for edu_inst in edu_inst_list if edu_inst is not None]
        if not edu_inst_list:
            return []
        else:
            return [html.unescape(inst) for inst in edu_inst_list if inst is not self.__empty_str and inst is not None]

    def get_company_names(self, label_set):
        company_list = label_set.findall("Jobs/job_company_name")
        company_list = [company.text for company in company_list]
        company_list = [self.replace_dash(company) for company in company_list if company is not None]
        if not company_list:
            return []
        else:
            return [html.unescape(company) for company in company_list if company is not self.__empty_str and company is not None]

    def get_job_titles(self, label_set):
        job_list = label_set.findall("Jobs/job_position")
        job_list = [job.text for job in job_list]
        job_list = [self.replace_dash(job) for job in job_list if job is not None]
        if not job_list:
            return []
        else:
            return [html.unescape(j_title) for j_title in job_list if j_title is not self.__empty_str and j_title is not None]

    def get_dataset_folder(self):
        return self.__dataset_raw_data_folder

    # file names examples, nr_of_file: -1 is limitless
    def populate_file_names(self, folder_path, nr_of_files=-1):
        filenames = []
        counter = 0
        for filename in os.listdir(folder_path):
            filename, file_ext = os.path.splitext(filename)
            filenames.append((filename, file_ext.lower()))
            counter += 1
            if counter == nr_of_files and nr_of_files != -1:
                break
        self.logger.println("read %s file names" % counter)
        return filenames

    def filter_by_valid_exts(self, filenames):
        valid_filenames = []
        counter = 0
        for filename_token in filenames:
            cur_ext = filename_token[1]
            if cur_ext == self.__file_ext_pdf or cur_ext == self.__file_ext_doc or cur_ext == self.__file_ext_docx:
                valid_filenames.append(filename_token)
        self.logger.println("filtered to %s valid file names" % counter)
        return valid_filenames


    def read_resume_content_tika_api(self, filenames, folder):
        os.environ['TIKA_VERSION'] = home + "1.14"
        os.environ['TIKA_SERVER_CLASSPATH'] = home + "tika-app-1.14.jar"

        remove_files_idxs = []
        resume_content = []
        for idx, filename in enumerate(filenames):
            self.logger.println("sending resume %s/%s to tika" % (idx+1, len(filenames)) )
            filepath = folder + self.__file_path_seperator + filename[0] + filename[1]
            extracted_information = parser.from_file(filepath)
            try:
                resume_content.append(extracted_information["content"])
            except KeyError:
                remove_files_idxs.append(idx)

        delete_count = 0
        for idx in remove_files_idxs:
            self.logger.println("removing unprocessed resume file at index %s named %s" % (idx-delete_count, filenames[idx-delete_count]))
            del filenames[idx-delete_count]
            delete_count += 1

        self.logger.println("removed %s files from internal data structure" % delete_count)
        self.logger.println("completed reading content from %s resume files" % len(resume_content))
        return filenames, resume_content

    def __read_resume_content(self):
        self.resume_content = []
        self.resume_metadata = []
        # idxs of files that don't have content
        remove_files_idxs = []

        Tika = autoclass('org.apache.tika.Tika')
        Metadata = autoclass('org.apache.tika.metadata.Metadata')
        FileInputStream = autoclass('java.io.FileInputStream')
        tika = Tika()
        meta = Metadata()

        for idx, filename in enumerate(self.dataset_filenames):
            self.logger.println("extracting from resume %s/%s with tika" % (idx, len(self.dataset_filenames)-1) )
            # append filename + ext to path
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + filename[1]
            try:
                extracted_information = tika.parseToString(FileInputStream(filepath), meta)
            except:
                extracted_information = ""

            # check if a supported file was processed successfully
            if len(extracted_information) > 0:
                self.resume_content.append(extracted_information)
            else:
                remove_files_idxs.append(idx)

        delete_count = 0
        for idx in remove_files_idxs:
            self.logger.println("removing unprocessed resume file at index %s named %s" % (idx-delete_count, self.dataset_filenames[idx-delete_count]))
            del self.dataset_filenames[idx-delete_count]
            delete_count += 1

        self.logger.println("read content from %s resume files" % len(self.resume_content))

    def read_resume_labels(self, folder, filenames):
        resume_labels = []
        for idx, filename in enumerate(filenames):
            filepath = folder + self.__file_path_seperator + filename[0] + self.__file_ext_xml
            xml_file = ET.ElementTree(file=filepath)
            resume_labels.append(xml_file)
        self.resume_labels = resume_labels
        self.logger.println("read labels from %s xml files" % len(self.resume_labels))
        return resume_labels

    def remove_empty_resumes(self, filenames, resume_content):
        if len(filenames) == len(resume_content):
            self.logger.println("filenames and résumé content structures same length OK")
        # idxs of files that don't have content
        remove_files_idxs = []
        for idx, file_content in enumerate(resume_content):
            if file_content is None:
                remove_files_idxs.append(idx)

        deleted_count = 0
        for idx in remove_files_idxs:
            self.logger.println("removing empty resume file at index %s named %s" % (idx, filenames[idx]))
            del filenames[idx-deleted_count]
            del resume_content[idx-deleted_count]
            deleted_count += 1
        self.logger.println("deleted %s files with no content" % deleted_count)
        self.logger.println("total file count is at %s" % len(resume_content))
        return filenames, resume_content

    # method is required when parsing and tagging a single file, mostly for
    # demonstration
    def read_resume_content(self, filepath):
        extracted_information = parser.from_file(filepath)
        docs = []
        docs.append(extracted_information["content"])
        return docs

    def read_resume_content_txtract(self):
        self.logger.println("extracting resume content using textract")
        self.resume_content = []
        # idxs of files that don't have content
        remove_files_idxs = []
        for idx, filename in enumerate(self.dataset_filenames):
            self.logger.println("extracting from resume %s/%s using txtract" % (idx+1, len(self.dataset_filenames)) )
            # append filename + ext to path
            filepath = self.__dataset_raw_data_folder + self.__file_path_seperator + filename[0] + filename[1]
            extracted_str = ""
            try:
                extracted_bytes = textract.process(filepath, encoding="utf_8")
                extracted_str = extracted_bytes.decode("utf-8")
                self.resume_content.append(extracted_str)
            except:
                self.logger.println("txtract threw an error")
                remove_files_idxs.append(idx)
        deleted_idxs = 0
        for idx in remove_files_idxs:
            self.logger.println("removing unprocessed resume file at index %s named %s" % (idx, self.dataset_filenames[idx]))
            del self.dataset_filenames[idx-deleted_idxs]

        self.logger.println("read content from %s resume files" % len(self.resume_content))

    def read_raw_files(self, nr_of_docs):
        """
        convenient method used to return xml and résumé files  
        """
        filenames = self.populate_file_names(self.__dataset_raw_data_folder, nr_of_docs)
        filenames = self.filter_by_valid_exts(filenames)
        filenames, resume_content = self.read_resume_content_tika_api(filenames, self.__dataset_raw_data_folder)
        filenames, resume_content = self.remove_empty_resumes(filenames, resume_content)
        resume_labels = self.read_resume_labels(self.__dataset_raw_data_folder, filenames)
        return resume_content, resume_labels

"""
ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
"""
