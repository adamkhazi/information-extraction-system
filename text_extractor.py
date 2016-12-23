import tika
import nltk

from tika import parser
from nltk.tokenize import word_tokenize

parsed = parser.from_file('test.pdf')
#print(parsed["metadata"])
#print(parsed["content"])

# type string
content = parsed["content"]

print("tokeninzing")

tknzr = word_tokenize(content)
#tokenized_content = tknzr.tokenize(content)
print(tknzr)

