from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy import en

from logger import Logger

# Class tokenises by words and lines
class Tokeniser():
    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("tokeniser created")
        #self.en_nlp = spacy.load('en')

    # Splits string content from résumé documents into lines based on \r and \n.
    def tokenise_docs_to_lines(self, docs):
        self.__logger.println("tokenising %s resumes by line" % len(docs))
        tokenised_docs = []
        for idx, file_content in enumerate(docs):
            tokenised_docs.append(file_content.splitlines())
        self.__logger.println("completed tokenising %s resumes by line" % len(docs))
        return tokenised_docs

    # Tokenises multiple strings (lines) into tokens/words 
    def tokenise_doclines_to_words(self, docs):
        self.__logger.println("tokenising %s resumes by words" % len(docs))
        tokenised_resumes = []
        for doc_idx, doc in enumerate(docs):
            tokenised_doc_lines = []
            for line_idx, line in enumerate(doc):
                tokens = self.tokenise_str(line)
                if tokens != []: # consider using spaces as features
                    filtered_words = self.remove_stop_words(tokens)
                    tokenised_doc_lines.append(filtered_words)
            tokenised_resumes.append(tokenised_doc_lines)
        self.__logger.println("completed tokenising %s resumes by words" % len(docs))
        return tokenised_resumes

    def docs_tolower(self, docs):
        self.__logger.println("lower casing %s resumes by tokens" % len(docs))
        return [[[token.lower() for token in line] for line in doc] for doc in docs]

    # Uses the regex tokeniser to tokenise a string by \w+ pattern
    def tokenise_str(self, content_str):
        rtokenizer = RegexpTokenizer(r'\w+')
        tokens = rtokenizer.tokenize(content_str)
        return tokens

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token.lower() not in en.STOP_WORDS]
