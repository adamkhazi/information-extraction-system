import pymssql
import nltk
import numpy
import os

from nltk.tokenize import word_tokenize
from numpy import array
from ast import literal_eval
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from database_connection import DatabaseConnection

conn = DatabaseConnection().connect()
cursor = conn.cursor()

#not random
#cursor.execute("SELECT TOP 50 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>10000 AND cn_res=0;")

#random
cursor.execute("SELECT TOP 200 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>10000 AND cn_res=0 ORDER BY NEWID();")

row = cursor.fetchone()

tw_temp = []
tw_tag_temp = []
pers_tag_count = 0
while row:
    rtokenizer = RegexpTokenizer(r'\w+')
    tokens = rtokenizer.tokenize(row[2])
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]

    temp = word_tokenize(" ".join(filtered_words))
    print("tagging: " + str(row[0]) + " and " + str(row[1]))
    for idx, n in enumerate(temp):
        names = word_tokenize((str(row[0]) + str(row[1])).lower())
        
        if any(n.lower() == s for s in names):
            tw_tag_temp.append("PERS")
            pers_tag_count +=1
        else: 
            tw_tag_temp.append("O")
    tw_temp = tw_temp + temp
    row = cursor.fetchone()
    
print("tokens: " + str(len(tw_temp)))
print("tags: " + str(len(tw_tag_temp)))
print("PERS tag count: " + str(pers_tag_count))

dataset_folder = "db_generated_datasets"
training_file = "ner_training_data.txt"
test_file = "ner_test_data.txt"

save_file = dataset_folder + "/" + test_file
try:
    os.remove(save_file)
except OSError:
    pass
with open (save_file,'a') as proc_seqf:
    for a, am in zip(tw_temp, tw_tag_temp):
        proc_seqf.write("{}\t{}\n".format(a, am))

print("saved: " + save_file)
