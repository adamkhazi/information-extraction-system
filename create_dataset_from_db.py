import pymssql
import nltk
import numpy
import os

from nltk.tokenize import word_tokenize
from numpy import array
from ast import literal_eval
from database_connection import DatabaseConnection

conn = DatabaseConnection().connect()
cursor = conn.cursor()

#not random
#cursor.execute("SELECT TOP 50 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>10000 AND cn_res=0;")

#random
cursor.execute("SELECT TOP 10 cn_fname, cn_lname, cn_resume FROM tblCandidate WHERE cn_fname IS NOT NULL AND DATALENGTH(cn_fname)>2 AND cn_lname IS NOT NULL AND DATALENGTH(cn_lname)>2 AND cn_resume LIKE '%[a-z0-9]%' AND DATALENGTH(cn_resume)>10000 AND cn_res=0 ORDER BY NEWID();")

row = cursor.fetchone()

tw_temp = []
tw_tag_temp = []
while row:
    temp = word_tokenize(row[2])
    print("testing for: " + str(row[0]) + " or " + str(row[1]))
    for idx, n in enumerate(temp):
        if ((len(n) >= len(str(row[0]))) and (n.lower() in str(row[0]).lower())) or ((len(n) >= len(str(row[1]))) and (str(row[1]).lower() in n.lower())):
            tw_tag_temp.append("PERS")
        else: 
            tw_tag_temp.append("O")
    tw_temp = tw_temp + temp
    row = cursor.fetchone()
    
print(len(tw_temp))
print(len(tw_tag_temp))

trainingfilename = "ner_training_data.txt"
testfilename = "ner_test_data.txt"

try:
    os.remove(trainingfilename)
except OSError:
    pass
with open (trainingfilename,'a') as proc_seqf:
    for a, am in zip(tw_temp, tw_tag_temp):
        proc_seqf.write("{}\t{}\n".format(a, am))
