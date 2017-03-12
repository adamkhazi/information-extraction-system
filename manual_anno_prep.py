from extractor import Extractor
from tokeniser import Tokeniser
from dataset import Dataset
import pdb

class ManualAnnoPrep():
    __manual_anno_folder = "manual_anno_data"
    __manual_anno_processed = "manual_anno_processed"

    def __init__(self):
        extractor = Extractor()
        filenames = extractor.populate_file_names(self.__manual_anno_folder)
        valid_filenames = extractor.filter_by_valid_exts(filenames)
        valid_filenames, resume_content = extractor.read_resume_content_tika_api(valid_filenames, self.__manual_anno_folder)

        tokeniser = Tokeniser()
        tokenised_docs = tokeniser.tokenise_docs_to_lines(resume_content)

        dataset = Dataset()
        dataset.save_doc_lines(tokenised_docs, valid_filenames, self.__manual_anno_processed)

if __name__ == "__main__":
    manual_anno = ManualAnnoPrep()
