import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np 
import tensorflow as tf
import random 
from main import load_data, sample, generate_text
import torch

name = ['base', 'base_droput', 'BiLSTM']
def main():
    text, char_to_index, index_to_char, characters = load_data()
    """ Names of saved models: [base, base_droput, BiLSTM] -- Change index as per need """
    model = tf.keras.models.load_model(f'Poetic_Text_Generation/Models/poetic_text_generator_{name[0]}.keras')

    # Get user input for the length and temperature
    length = int(input("Enter the length of text to generate: "))
    temperature = float(input("Enter the temperature for text generation (e.g., 0.5, 1.0, 1.5): "))

    generated_text = generate_text(model, text, char_to_index, index_to_char, characters, length, temperature)
    print("Generated Text:\n", generated_text)

if __name__ == "__main__":
    main()
