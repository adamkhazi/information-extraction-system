import pymssql
import nltk
import numpy
import os
import io
import sys, time

from nltk.tokenize import word_tokenize
from numpy import array
from ast import literal_eval
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

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
            ("select TOP " + str(doc_nr) + " cn_fname, cn_lname, cn_resume " # not cn_res & not random
           "FROM tblCandidate "
           "WHERE cn_fname IS NOT NULL "
           "AND DATALENGTH(cn_fname)>2 "
           "AND cn_lname IS NOT NULL "
           "AND DATALENGTH(cn_lname)>2 "
           "AND cn_resume LIKE '%[a-z0-9]%' "
           "AND DATALENGTH(cn_resume)>17000;")
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
        for row in self.__db_cursor:
            self.raw_db_table.append(row)
        print("Pulled " + str(len(self.raw_db_table)) + " records")

    def tokenize_text(self):
        self.tokenized_text = []
        for doc in self.raw_db_table:
            #rtokenizer = RegexpTokenizer(r'\w+')
            #tokens = rtokenizer.tokenize(doc[2])
            tokens = word_tokenize(doc[2])
            #optional remove stop words
            #filtered_words = [w for w in tokens if not w in stopwords.words('english')]
            self.tokenized_text.append(tokens)
        print("Tokenized text")
    
    def pos_tag_tokens(self):
        self.pos_doc_tokens = []
        for doc in self.tokenized_text:
            tagged_tokens = nltk.pos_tag(doc)
            self.pos_doc_tokens.append(tagged_tokens)
        print("POS tagged tokens")
            
    def ner_tag_tokens(self):
        self.ner_doc_tokens = []
        pers_count = 0

        for doc_idx, doc in enumerate(self.tokenized_text):
            single_ner_doc = []
            for token_idx, token in enumerate(doc):
                rtokenizer = RegexpTokenizer(r'\w+')
                matching_names = rtokenizer.tokenize((str(self.raw_db_table[doc_idx][0]) + " " + str(self.raw_db_table[doc_idx][1])).lower())
                
                if any(token.lower() == s for s in matching_names):
                    #replace word with tagged tuple
                    single_ner_doc.append((token, "PERS"))
                    pers_count += 1
                else: 
                    single_ner_doc.append((token, "O"))

            self.ner_doc_tokens.append(single_ner_doc)
        print("NER tagged tokens")

    def save_tagged_tokens(self):
        path = self.__dataset_folder + "/" + self.__dataset_name
        try:
            os.remove(path)
        except OSError:
            pass
        with io.open (path,'a', encoding='utf-8') as proc_seqf:
            for doc_idx, doc in enumerate(self.ner_doc_tokens):
                for token_idx, token in enumerate(doc):
                    # token, pos_tag, ner_tag
                    proc_seqf.write("{}\t{}\t{}\n".format(token[0], self.pos_doc_tokens[doc_idx][token_idx][1], token[1]))
                proc_seqf.write("\n")

        print("Saved tagged tokens to: " + path)
