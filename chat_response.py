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
from chatmodel import words_presence

with open('intents.json') as data:
    intents = json.load(data)
    
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

#Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
try:
    model.load('model.tflearn')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")


context = {}
ERROR_THRESHOLD = 0.25

def classify(sentence):
    results = model.predict([words_presence(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": r[1]} for r in results]

bot_name = "Bakerly"
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        for result in results:
            for intent in intents['intents']:
                if intent['tag'] == result['intent']:
                    if 'context_set' in intent:
                        context[userID] = intent['context_set']
                    if 'context_filter' not in intent or (userID in context and context[userID] == intent['context_filter']):
                        return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

if __name__ == '__main__':
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "How are you?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = response(sentence)
        print(bot_name,":",resp)
