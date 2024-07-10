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
clases = pickle.load(open('Intelligent_AI_Chatbot/Pickle_Files/classes.pkl', 'rb'))
model = load_model('Intelligent_AI_Chatbot/Models/Chat_Bot_Model.keras')

# Working Functions
def clean_up_sentence(sentence):
    pass