import random
import pickle
import json
import numpy as np
import tensorflow as tf
import nltk
import keras
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("../intents.json").read())

words = []
documents = []
classes = []
ignores = ['!',',','.','?','/']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignores]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
outputEmpty = [0]*len(classes)

for document in documents:
    bag = []
    wordPattens = document[0]
    wordPattens = [lemmatizer.lemmatize(word.lower()) for word in wordPattens]
    for word in words: bag.append(1) if word in wordPattens else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag+outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:,:len(words)]
trainY = training[:,len(words):]

model = keras.Sequential()
model.add(keras.layers.Dense(128,input_shape=(len(trainX[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(8,activation='softmax'))

sgd = keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX),np.array(trainY), epochs=200, batch_size=5,verbose=1)
model.save('echobot_model.keras',hist)
with open('training_history.pkl', 'wb') as history_file:
    pickle.dump(hist.history, history_file)
print('Done')

