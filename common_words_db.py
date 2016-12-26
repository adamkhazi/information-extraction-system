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

#cursor.execute("select cn_fname, count(cn_fname) c from tblCandidate group by cn_fname order by c desc;")
#cursor.execute("select top 2000 cn_fname, cn_lname from tblCandidate where cn_res=0;")
cursor.execute("select COUNT(*) from tblCandidate where cn_res=0;")

row = cursor.fetchone()

while row:
    #print("firstname: " + str(row[0]) + " lastname:" + str(row[1]))
    print(row[0])
    row = cursor.fetchone()
