#Importing Libraries
import nltk
from nltk.stem import WordNetLemmatizer
from PIL import Image as pil
import tflearn
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import random
import json
import pickle
from flask import Flask, request, jsonify

stemmer = WordNetLemmatizer()

with open('intents.json') as data:
    intents = json.load(data)

words = []
classes = []
documents = []
ignore_words = ['?','.','!',',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Tokenizing each word
        w = nltk.word_tokenize(pattern,language="english", preserve_line=False)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([stemmer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(list(set(classes)))

#Creating Training data
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#Shuffling features and turning into array
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = [entry[0] for entry in training]
train_y = [entry[1] for entry in training]

train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)

#print(train_x[0])
#print(train_y[0])

# reset underlying graph data
tf.compat.v1.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=16, show_metric=True)

model.save('model.tflearn')

#Pickling data
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

#Importing chatbot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)
    
#Unpickling data
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']

train_x = data['train_x']
train_y = data['train_y']

tf.compat.v1.reset_default_graph()

#Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

def clean_up_sentence(sentence):
    #Tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    #Stemming each word
    sentence_words = [stemmer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return: 0(absence) or 1(presence) for each word in the bag that exists in the sentence
def words_presence(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1

    return(np.array(bag))