import tika
from tika import parser
parsed = parser.from_file('test.pdf')
print(parsed["metadata"])
print(parsed["content"])
