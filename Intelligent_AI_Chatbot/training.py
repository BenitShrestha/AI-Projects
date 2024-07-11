import random 
import json 
import pickle 
import numpy as np 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import nltk # Natural Language Toolkit, downloaded punkt and wordnet
from nltk.stem import WordNetLemmatizer # Reduces words to its stem -- i.e. 'works', 'working' -> 'work'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Create a lemmatizer
lemmatizer = WordNetLemmatizer()

# Load JSON file
intents = json.loads(open('Intelligent_AI_Chatbot/JSON_Files/intents.json').read())

words = [] # Keeps tokenized words from pattern
classes = [] # Keep tags
documents = [] # Keeps words and tags
ignore_letters = ['?','!', '.', ',']

for intent in intents['intents']: # Key intents - Basically a dictionary
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # Tokenize words - Splits sentence into list of words, returns a list as well
        words.extend(word_list) # Extend: Takes content of lists and append to list-- Append: Lists appended to list
        documents.append((word_list, intent['tag'])) # Appended word list belongs to respective tag
        """ In above code, tuple is passed """
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(words) # Includes repeated as well as unique words
# print(documents)

""" Edit words list, by lemmatizing words -- i.e. 'works', 'working' -> 'work' """
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] # First lemmatize done for saving in pickle file

words = sorted(set(words)) # Sorted and unique words
classes = sorted(set(classes))

# print(words) # Lemmatized words, unique and sorted list
# print(classes) # Unique tags

pickle.dump(words, open('Intelligent_AI_Chatbot/Pickle_Files/words.pkl', 'wb')) # For serialization, deserialization -- Objects, types to bytes
pickle.dump(classes, open('Intelligent_AI_Chatbot/Pickle_Files/classes.pkl', 'wb'))

""" Bag of Words - 0s and 1s: If word is present, 1, else 0 """
training = []
output_empty = [0] * len(classes) # List of zeros with length of classes

# Preparing training data
for document in documents: # Document consists of words from patterns and tags
    bag = []
    word_patterns = document[0] # Accessing words from patterns
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # Second Lemmatize for bag of words probabaly
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle, randomize data
random.shuffle(training)
training = np.array(training, dtype=object)  # Specify dtype=object to handle lists of different lengths

train_x = np.array([item[0] for item in training], dtype=np.float32)  # Convert each bag to a numpy array
train_y = np.array([item[1] for item in training], dtype=np.float32)  # Convert each output_row to a numpy array

# Building Model
model = Sequential()
model.add(Dense(256, input_shape = (len(train_x[0]), ) , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.25))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

# Optimizer, loss function and compilation
sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Fitting
model.fit(train_x, train_y, epochs = 300, batch_size = 5, verbose = 1)

model.save('Intelligent_AI_Chatbot/Models/Chat_Bot_Model_normal.keras')

print("Done!")