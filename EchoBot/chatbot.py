import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.api.models import load_model

from EchoBot.new import words, model

lemmatizer = WordNetLemmatizer(0)
intents = json.loads(open('../intents.json').read())
words = pickle.loads(open('words.pkl','rb'))
classes = pickle.loads(open('classes.pkl','rb'))
model = load_model('echobot_model.h5')

def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = cleanup_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)