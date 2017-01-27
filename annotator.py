from extractor import Extractor
from tagger import Tagger
from dataset import Dataset
from tokeniser import Tokeniser

# Class is used to extract data from résumé documents,
# annotate documents and save to disk
class Annotator():
    __job_position_tag = "EMP-POS"
    __job_company_tag = "EMP-COMP"

    __education_course_tag = "EDU-MAJOR"
    __education_institution_tag = "EDU-INST"

    def __init__(self):
        self.__extractor = Extractor()
        self.__tokeniser = Tokeniser()
        self.__tagger = Tagger()
        self.__dataset = Dataset()

    def prepare_dataset(self, nr_of_docs=-1):
        resumes, labels = self.__extractor.read_raw_files(nr_of_docs)

        resumes = self.__tokeniser.tokenise_docs_to_lines(resumes)
        resumes = self.__tokeniser.tokenise_doclines_to_words(resumes)

        self.__dataset.resume_content = self.annotate_docs(resumes, labels)
        self.__dataset.save()

    # resumes: list of tokenised (by line and word) résumé docs
    # labels: xml structure storing labels for several resumes
    def annotate_docs(self, resumes, labels):
        annotated_resumes = []
        for idx, resume in enumerate(resumes):
            annotated_resumes.append(self.annotate_doc(resume, labels[idx]))
        return annotated_resumes

    # doc: a single résumé document with token strings in each slot of list
    # labels: xml structure storing pre-extracted information
    def annotate_doc(self, doc, labels):
        job_title_list = self.__extractor.get_job_titles(labels)
        job_company_list = self.__extractor.get_company_names(labels)
        edu_major_list = self.__extractor.get_edu_majors(labels)
        edu_inst_list = self.__extractor.get_edu_institutions(labels)
        # can extract more labels here

        prepared_doc = self.__tagger.prepare_doc(doc)
        prepared_doc = self.__match_entity(prepared_doc, job_title_list, self.__job_position_tag)
        prepared_doc = self.__match_entity(prepared_doc, job_company_list, self.__job_company_tag)
        prepared_doc = self.__match_entity(prepared_doc, edu_major_list, self.__education_course_tag)
        prepared_doc = self.__match_entity(prepared_doc, edu_inst_list, self.__education_institution_tag)

        prepared_doc = self.__tagger.pos_tag(prepared_doc)
        prepared_doc = self.__tagger.nonlocal_ne_tag(prepared_doc)

        return prepared_doc

    # doc: résumé doc to be annotated
    # entity_list: list of labels to matched in doc
    # tag: tag to be assigned if match found
    def __match_entity(self, doc, entity_list, tag):
        for entity in entity_list:
            doc = self.__tagger.match_label(doc, entity, tag)
        return doc

    # function takes in a path to file and annotates it for tagging
    # to be ideally used to tag as a one off for testing
    # filepath: path to résumé 
    def annotate_using_trained_model(self, filepath):
        resume_content = self.__extractor.read_resume_content(filepath)

        resume_content = self.__tokeniser.tokenise_docs_to_lines(resume_content)
        resume_content = self.__tokeniser.tokenise_doclines_to_words(resume_content)

        prepared_doc = self.__tagger.prepare_doc(resume_content[0])
        prepared_doc = self.__tagger.pos_tag(prepared_doc)

        return prepared_doc
