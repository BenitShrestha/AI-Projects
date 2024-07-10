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

# print(words) # Includes repeated as well as unique words
# print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

# print(words) # Lemmatized words, unique and sorted list
# print(classes) # Unique tags

pickle.dump(words, open('Intelligent_AI_Chatbot/Pickle Files/words.pkl', 'wb'))
pickle.dump(classes, open('Intelligent_AI_Chatbot/Pickle Files/classes.pkl', 'wb'))

""" Bag of Words - 0s and 1s: If word is present, 1, else 0 """
training = []
output_empty = [0] * len(classes) # List of zeros with length of classes

# Preparing training data
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
 
# Shuffle, randomize data
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Building Model
model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]), ) , activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

# Optimizer, loss function and compilation
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Fitting
model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)

model.save('Intelligent_AI_Chatbot/Models/Chat_Bot_Model.keras')

print("Done!")