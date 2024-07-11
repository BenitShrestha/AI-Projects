# Start -- Using neural networks and Natural language processing
# Statically typed into a JSON file, has static response to user input

# Chatbot
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random 
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Intialization of JSON file, Pickle files and Model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('Intelligent_AI_Chatbot/JSON_Files/intents.json').read())

words = pickle.load(open('Intelligent_AI_Chatbot/Pickle_Files/words.pkl', 'rb'))
classes = pickle.load(open('Intelligent_AI_Chatbot/Pickle_Files/classes.pkl', 'rb'))
model = load_model('Intelligent_AI_Chatbot/Models/Chat_Bot_Model_normal.keras')

# Working Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # First tokenize
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # Second Lemmatize
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence) # Create a bag of words
    res = model.predict(np.array([bow]))[0] # Predict based on bag of words
    ERROR_THRESHOLD = 0.25 # 25% threshold for uncertainty
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # Get index of class and probabilites too
    
    results.sort(key = lambda x: x[1], reverse = True) # Sort based on probability, high to low
    return_list =[] 
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])}) # Return list full of intents/classes and probability
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Sorry, I didn't get that. Please try again."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Chatbot is running!')

while True:
    message = input("")
    ints = predict_class(message)
    if ints: # Check if prediction returned any intents
        res = get_response(ints, intents)
    else:
        res = "Sorry I got confused"
    print(res)