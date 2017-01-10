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

conn = DbConnection().connect()
cursor = conn.cursor()

sql_query_ordered = "SELECT TOP 50 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>10000 AND cn_res=0;"

sql_query_random = "SELECT TOP 3000 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>17000 AND cn_res=0 ORDER BY NEWID();"

cursor.execute(sql_query_random)

row = cursor.fetchone()

tagged_tokens = []
pers_count = 0

while row:
    rtokenizer = RegexpTokenizer(r'\w+')
    tokens = rtokenizer.tokenize(row[2])
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]

    temp_doc_tokens = word_tokenize(" ".join(filtered_words))
    temp_pos_tags = nltk.pos_tag(temp_doc_tokens)
    
    print('\rTagging document nr: ' + str(len(tagged_tokens)), end='')

    for idx, n in enumerate(temp_doc_tokens):
        names = rtokenizer.tokenize((str(row[0]) + " " + str(row[1])).lower())
        
        if any(n.lower() == s for s in names):
            #replace word with tagged tuple
            temp_doc_tokens[idx] = (temp_doc_tokens[idx], temp_pos_tags[idx][1], "PERS")
            pers_count += 1
        else: 
            temp_doc_tokens[idx] = (temp_doc_tokens[idx], temp_pos_tags[idx][1], "O")

    row = cursor.fetchone()
    # each list within tagged_tokens is a document
    tagged_tokens.append(temp_doc_tokens)
    
print("documents: " + str(len(tagged_tokens)))
print("tokens: " + str(sum([len(doc) for doc in tagged_tokens])))
# TODO PERS COUNT
#print("PERS count: " + str(sum(a in doc for doc in tagged_tokens)))

dataset_folder = "db_generated_datasets"
ner_file = "ner_dataset.txt"

save_file = dataset_folder + "/" + ner_file
try:
    os.remove(save_file)
except OSError:
    pass
with io.open (save_file,'a', encoding='utf-8') as proc_seqf:
    for idx, item in enumerate(tagged_tokens):
        for token in enumerate(tagged_tokens[idx]):
            proc_seqf.write("{}\t{}\t{}\n".format(token[1][0], token[1][1], token[1][2]))
        proc_seqf.write("\n")

print("saved to: " + save_file)
