# Start -- Using neural networks and Natural language processing
# Statically typed into a JSON file, has static response to user input

# Chatbot

import random 
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Intialization of JSON file, Pickle files and Model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Intelligent_AI_Chatbot/intents.json').read())

words = pickle.load(open('Intelligent_AI_Chatbot/Pickle_Files/words.pkl', 'rb'))
classes = pickle.load(open('Intelligent_AI_Chatbot/Pickle_Files/classes.pkl', 'rb'))
model = load_model('Intelligent_AI_Chatbot/Models/Chat_Bot_Model.keras')

# Working Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # First tokenize
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # Second Lemmatize
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    pass