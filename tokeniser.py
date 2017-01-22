from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Tokeniser():
    def tokenise_docs_to_lines(self, docs):
        tokenised_docs = []
        for idx, file_content in enumerate(docs):
            tokenised_docs.append(file_content.splitlines())
        return tokenised_docs

    def tokenise_doclines_to_words(self, docs):
        tokenised_resumes = []
        rtokenizer = RegexpTokenizer(r'\w+')
        for doc_idx, doc in enumerate(docs):
            tokenised_doc_lines = []
            for line_idx, line in enumerate(doc):
                line = rtokenizer.tokenize(line)
                if line != []: # consider using spaces as features
                    filtered_words = [word for word in line if word not in stopwords.words('english')]
                    tokenised_doc_lines.append(filtered_words)
            tokenised_resumes.append(tokenised_doc_lines)
        return tokenised_resumes

