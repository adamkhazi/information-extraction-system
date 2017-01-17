import pymssql
import nltk
import numpy
import os
import io
import csv
import sys, time
import glob

from nltk.tokenize import word_tokenize, sent_tokenize
from numpy import array
from ast import literal_eval
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.internals import find_jars_within_path
from os.path import expanduser

from db_connection import DbConnection

class GenerateDataset:
    __dataset_folder = "db_generated_datasets"
    __dataset_name = "ner_dataset.txt"

    def __concatenate_sql_queries_and_select(self, doc_nr, query_nr):
        # nr 2 is ordered randomly
        self.sql_query_list = [
            ("select TOP " + str(doc_nr) + " cn_fname, cn_lname, cn_resume " # is cn_res not random
           "from tblCandidate "
           "where cn_fname IS NOT NULL "
           "AND DATALENGTH(cn_fname)>2 "
           "AND cn_lname IS NOT NULL "
           "AND DATALENGTH(cn_lname)>2 "
           "AND cn_resume LIKE '%[a-z0-9]%' "
           "AND DATALENGTH(cn_resume)>14000 "
           "AND cn_res=0;"),
            ("SELECT TOP " + str(doc_nr) + " cn_fname, cn_lname, cn_resume " # is random
           "FROM tblCandidate "
           "WHERE cn_fname IS NOT NULL "
           "AND DATALENGTH(cn_fname)>2 "
           "AND cn_lname IS NOT NULL "
           "AND DATALENGTH(cn_lname)>2 "
           "AND cn_resume LIKE '%[a-z0-9]%' "
           "AND DATALENGTH(cn_resume)>17000 "
           "AND cn_res=0 ORDER BY NEWID();"),
            ("select TOP " + str(doc_nr) + " cn_fname, cn_lname, cn_resume, cn_present_position "  # not cn_res & not random
           "FROM tblCandidate "
           "WHERE cn_fname IS NOT NULL "
           "AND DATALENGTH(cn_fname)>2 "
           "AND cn_lname IS NOT NULL "
           "AND DATALENGTH(cn_lname)>2 "
           "AND cn_resume LIKE '%[a-z0-9]%' "
           "AND DATALENGTH(cn_resume)>10000 "
           "AND cn_present_position IS NOT NULL "
           "AND cn_present_position LIKE '%[a-z0-9]%' "
           "AND cn_res=0 ORDER BY NEWID();")
        ]
        return self.sql_query_list[query_nr]

    def __get_db_cursor(self):
        self.__db_cursor = DbConnection().connect()

    def __set_sql_query(self, query):
        self.__query_to_execute = query

    def __execute_query(self):
        self.__db_cursor.execute(self.__query_to_execute)

    def pull_db_records(self, query_nr, doc_nr):
        self.__get_db_cursor()
        self.__set_sql_query(self.__concatenate_sql_queries_and_select(doc_nr, query_nr))
        self.__execute_query()

        self.raw_db_table = []
        print("Pulling " + str(doc_nr) + " records")
        self.raw_db_table = self.__db_cursor.fetchall()
        print("Pulled " + str(len(self.raw_db_table)) + " records")

    def tokenize_text(self):
        self.tokenized_docs_by_lines = []
        for doc in self.raw_db_table:
            #rtokenizer = RegexpTokenizer(r'\w+')
            #tokens = rtokenizer.tokenize(doc[2])
            doc_lines = doc[2].splitlines()

            tokenized_doc_lines = []
            rtokenizer = RegexpTokenizer(r'\w+')
            for line in doc_lines:
                line = rtokenizer.tokenize(line)
                #line = word_tokenize(line)

                if line != []:
                    tokenized_doc_lines.append(line)

            #optional remove stop words
            #filtered_words = [w for w in tokens if not w in stopwords.words('english')]

            # append doc to global list
            self.tokenized_docs_by_lines.append(tokenized_doc_lines)

        print("Split lines and tokenized text")

    def pos_tag_tokens(self):
        self.pos_doc_tokens = []
        for doc in self.tokenized_docs_by_lines:

            tagged_doc_lines = []
            print("done pos tag for doc")
            for line in doc:
                tagged_line = nltk.pos_tag(line)
                tagged_doc_lines.append(tagged_line)

            self.pos_doc_tokens.append(tagged_doc_lines)
        print("POS tagged tokens")

    def ner_tag_tokens(self):
        self.name_tag_tokens()
        self.current_position_tag_tokens()

    def name_tag_tokens(self):
        self.ner_doc_tokens = []

        for doc_idx, doc in enumerate(self.tokenized_docs_by_lines):
            tagged_doc = []

            for line in doc:
                single_doc_line = []
                for token_idx, token in enumerate(line):
                    rtokenizer = RegexpTokenizer(r'\w+')
                    matching_names = rtokenizer.tokenize((str(self.raw_db_table[doc_idx][0]) + " " + str(self.raw_db_table[doc_idx][1])).lower())

                    if any(token.lower() == s for s in matching_names):
                        #replace word with tagged tuple
                        single_doc_line.append((token, "PERS"))
                    else:
                        single_doc_line.append((token, "O"))

                tagged_doc.append(single_doc_line)

            self.ner_doc_tokens.append(tagged_doc)

        print("NER name tagged tokens")

    def current_position_tag_tokens(self):
        for doc_idx, doc in enumerate(self.tokenized_docs_by_lines):
            for line_idx, line in enumerate(doc):
                matching_curpos_window = word_tokenize((str(self.raw_db_table[doc_idx][3])).lower())
                last_index_of_line = len(line)-1
                last_index_of_window = len(matching_curpos_window)-1
                for current_tkn_idx in range(0, (last_index_of_line - last_index_of_window) + 1):
                    current_window = line[current_tkn_idx:current_tkn_idx+len(matching_curpos_window)]
                    current_window = [x.lower() for x in current_window]
                    if current_window == matching_curpos_window:
                        # change ner tag to current position
                        for found_idx in range(current_tkn_idx, (current_tkn_idx + last_index_of_window) + 1):
                            self.ner_doc_tokens[doc_idx][line_idx][found_idx] = (line[found_idx], "EMPHIST-CURPOS")

        print("NER current position tagged tokens")

    def nonlocal_ner_tag_tokens(self):
        home = expanduser("~")
        os.environ['CLASSPATH'] = home + '/stanford-ner-2015-12-09'
        os.environ['STANFORD_MODELS'] = home + '/stanford-ner-2015-12-09/classifiers'

        st = StanfordNERTagger("english.all.3class.distsim.crf.ser.gz", java_options='-mx4000m')

        stanford_dir = st._stanford_jar[0].rpartition('/')[0]
        stanford_jars = find_jars_within_path(stanford_dir)

        st._stanford_jar = ':'.join(stanford_jars)

        # do not tokenise text
        nltk.internals.config_java(options='-tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions "tokenizeNLs=true"')

        self.nonlocal_ner_doc_tokens = []
        temp_nonlocal_bulk_process = []
        length_of_docs = [len(doc) for doc in self.tokenized_docs_by_lines]
        for doc_idx, doc in enumerate(self.tokenized_docs_by_lines):
            for line_idx, line in enumerate(doc):
                temp_nonlocal_bulk_process.append(line)

        temp_nonlocal_bulk_process = st.tag_sents(temp_nonlocal_bulk_process)

        current_idx = 0
        for doc_len_idx, doc_len in enumerate(length_of_docs):
            self.nonlocal_ner_doc_tokens.append(temp_nonlocal_bulk_process[current_idx:current_idx+doc_len])
            current_idx += doc_len
        print("NER nonlocal tagged tokens")

    def save_tagged_tokens(self):
        directory=self.__dataset_folder
        files=glob.glob('*.txt')
        for filename in files:
            os.unlink(filename)

        path = self.__dataset_folder + "/"

        for doc_idx, doc in enumerate(self.ner_doc_tokens):
            doc_file = open(path + str(doc_idx) + '.txt', 'w', encoding='utf-8')
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    # token, pos_tag, ner_tag
                    #print("length ner doc: " + str(len(doc)) + " length pos doc: " + str(len(self.pos_doc_tokens[doc_idx])) + " length nonlocal doc: " + str(len(self.nonlocal_ner_doc_tokens[doc_idx])))
                    doc_file.write("{}\t{}\t{}\t{}\n".format(token[0], self.pos_doc_tokens[doc_idx][line_idx][token_idx][1], self.nonlocal_ner_doc_tokens[doc_idx][line_idx][token_idx][1], token[1]))
                doc_file.write("\n")
            doc_file.close()

        print("Saved tagged tokens to: " + path)

    def read_tagged_tokens(self):
        dataset_docs = []
        for filename in os.listdir(self.__dataset_folder):
            current_file_path = self.__dataset_folder + "/" + filename
            if current_file_path.endswith(".txt"):
                with io.open(current_file_path, 'r', encoding='utf-8') as tsvin:
                    single_doc = []
                    single_line = []
                    tsvin = csv.reader(tsvin, delimiter='\t')
                    for row in tsvin:
                        if not row:
                            single_doc.append(single_line)
                            single_line = []
                        else:
                            single_line.append((row[0], row[1], row[2], row[3]))
                    dataset_docs.append(single_doc)
        return dataset_docs

