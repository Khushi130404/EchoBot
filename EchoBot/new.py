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