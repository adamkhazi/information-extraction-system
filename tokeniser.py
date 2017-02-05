from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from logger import Logger

import pdb

class Tokeniser():
    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("tokeniser created")

    def tokenise_docs_to_lines(self, docs):
        self.__logger.println("tokenising %s resumes by line" % len(docs))
        tokenised_docs = []
        for idx, file_content in enumerate(docs):
            self.__logger.println("tokenising resume nr %s" % idx)
            tokenised_docs.append(file_content.splitlines())
        self.__logger.println("completed tokenising %s resumes by line" % len(docs))
        return tokenised_docs

    def tokenise_doclines_to_words(self, docs):
        tokenised_resumes = []
        for doc_idx, doc in enumerate(docs):
            tokenised_doc_lines = []
            for line_idx, line in enumerate(doc):
                line = word_tokenize(line)
                if line != []: # consider using spaces as features
                    filtered_words = [word for word in line if word not in stopwords.words('english')]
                    tokenised_doc_lines.append(filtered_words)
            tokenised_resumes.append(tokenised_doc_lines)
        return tokenised_resumes

