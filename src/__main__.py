from chunker import chunk_python
from tokenizer import indexer

code = """
import os

def hello(name):
    print(name)

class Person:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(self.name)
"""

chunks = []
index_dict = {}
chunks.extend(chunk_python("here.py", code, 2000))
for id, chunk in enumerate(chunks):
	indexer(chunk["text"], id, index_dict)

for text, chunks in index_dict.items():
	print("'", text, "' :", chunks)
