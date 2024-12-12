import random
import pickle
import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = []
documemts = []
classes = []
ignores = ['!',',','.','?','/']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        documemts.append(wordList,intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(words) for word in words if word not in ignores]
words = sorted(set(classes))
classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
