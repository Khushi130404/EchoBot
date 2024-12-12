import random
import pickle
import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.utils.version_utils import training

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

training = []
outputEmpty = [0]*len(classes)

for documemt in documemts:
    bag = []
    wordPattens = documemt[0]
    wordPattens = [lemmatizer.lemmatize(word.lower()) for word in wordPattens]
    for word in words: bag.append(1) if word in wordPattens else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.indexDocument[1]] = 1
    training.append(bag+outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:,:len(words)]
trainY = training[:,len(words):]

