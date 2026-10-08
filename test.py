import ast

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

tree = ast.parse(code)

for node in ast.iter_child_nodes(tree):
    print(type(node))
