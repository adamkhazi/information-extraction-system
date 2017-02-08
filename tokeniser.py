#from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy import en

from logger import Logger
import pdb

# Class tokenises by words and lines
class Tokeniser():
    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("tokeniser created")
        self.en_nlp = spacy.load('en')

    def tokenise_docs_to_lines(self, docs):
        self.__logger.println("tokenising %s resumes by line" % len(docs))
        tokenised_docs = []
        for idx, file_content in enumerate(docs):
            tokenised_docs.append(file_content.splitlines())
        self.__logger.println("completed tokenising %s resumes by line" % len(docs))
        return tokenised_docs

    def tokenise_doclines_to_words(self, docs):
        self.__logger.println("tokenising %s resumes by words" % len(docs))
        tokenised_resumes = []
        for doc_idx, doc in enumerate(docs):
            tokenised_doc_lines = []
            for line_idx, line in enumerate(doc):
                tokens = word_tokenize(line)
                if tokens != []: # consider using spaces as features
                    filtered_words = [token for token in tokens if token.lower() not in en.STOP_WORDS]
                    tokenised_doc_lines.append(filtered_words)
            tokenised_resumes.append(tokenised_doc_lines)
        self.__logger.println("completed tokenising %s resumes by words" % len(docs))
        return tokenised_resumes

