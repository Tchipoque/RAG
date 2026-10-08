import tokenize
from io import BytesIO

def tokenizer(content) :
	tokens = []
	frase = ""
	for word in content:
		if not word.isspace():
			frase += word
		elif word.isspace() and frase:
			tokens.append(frase)
			frase = ""

	return tokens

import tokenize
from io import BytesIO

def indexer(text, chunk_id, index_dict):
    tokens = tokenize.tokenize(BytesIO(text.encode("utf-8")).readline)
    for token in tokens:
        word = token.string
        # skip empty/whitespace tokens (ENCODING, NEWLINE, INDENT, DEDENT, ENDMARKER...)
        if not word.strip() or token.type == tokenize.ENCODING:
            continue
        inner = index_dict.setdefault(word, {})
        inner[chunk_id] = inner.get(chunk_id, 0) + 1





code = """
import os

def hello(name):
    print(name)

class Person:
    def __init__(self, name):
        self.name = name
"""



