import tika
import nltk
import numpy

from tika import parser
from nltk.tokenize import word_tokenize

parsed = parser.from_file('test.pdf')
#print(parsed["metadata"])
#print(parsed["content"])

# type string
content = parsed["content"]

print("tokeninzing")

tokenized_words = word_tokenize(content)
#tokenized_content = tknzr.tokenize(content)

pos_words = nltk.pos_tag(tokenized_words)

ner_words = nltk.ne_chunk(pos_words)
print(ner_words)
