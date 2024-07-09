import random 
import json 
import pickle 
import numpy as np 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk # Natural Language Toolki, downloaded punkt and wordnet
from nltk.stem import WordNetLemmatizer # Reduces words to its stem -- i.e. 'works', 'working' -> 'work'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Create a lemmatizer
lemmatizer = WordNetLemmatizer()

# Load JSON file
intents = json.loads(open('Intelligent_AI_Chatbot/intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!', '.', ',']

for intent in intents['intents']: # Key intents - Basically a dictionary
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # Tokenize words - Splits sentence into list of words, returns a list as well
        words.extend(word_list) # Extend: Takes content of lists and append to list-- Append: Lists appended to list
        documents.append((word_list, intent['tag'])) # Appended word list belongs to respective tag
        """ In above code, tuple is passed """
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(words) # Includes repeated as well as unique words
# print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words) # Lemmatized words, unique and sorted list