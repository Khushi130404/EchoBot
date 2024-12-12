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
