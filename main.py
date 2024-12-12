import numpy
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stammer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

print(data)