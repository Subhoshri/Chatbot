#Importing Libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
import numpy as np
import tflearn
import random
import json

stemmer = LancasterStemmer()

with open('intents.json') as data:
    ints = json.load(data)

words = []
classes = []
documents = []
ignore = ['?']
for intent in ints['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern) #tokenize each words
        words.extend(w) #add to word list
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print('Documents: ',len(documents))
print('Classes: ',len(classes),classes)
print(len(words),"Stemmed Words",words)


