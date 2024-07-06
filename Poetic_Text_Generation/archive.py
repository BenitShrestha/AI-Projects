'''
import numpy as np 
import tensorflow as tf
import random 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

def main():
    """ Loading data from Gutenberg -- Shakespeare Texts, then loading to device in read-binary mode and decoding the text into lowercase """
    filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://www.gutenberg.org/files/100/100-0.txt')

    text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

    text = text[30000:80000] # Slicing to reduce actual data size 

    characters = sorted(set(text)) # Unique characters using Set 

    """ Mapping characters to index and index to characters using a Dictionary: {'a': 1, 'f': 7} & {1: 'a', 7: 'f'} """
    char_to_index = dict((c, i) for i, c in enumerate(characters))
    index_to_char = dict((i, c) for i, c in enumerate(characters))

    SEQ_LENGTH = 40 # Set sequence length for next character prediction 
    STEP_SIZE = 3 # Samples of sequence taken after every 3 characters -- Hello, this is Benit : lo, 

    """ To store sentences and generate next characters, example: The Sky is Blu e """
    sentences = []
    next_characters = []

    """ Loop range ensures you don't start too close to end of the text"""
    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE): 
        sentences.append(text[i: i + SEQ_LENGTH]) # Append to sentences with certain sequence length 
        next_characters.append(text[i + SEQ_LENGTH]) # Append next_character to sentence, +1 to sequence length

    """ Now to convert these characters into Numpy Array """
    x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool) # Input
    y = np.zeros((len(sentences), len(characters)), dtype=bool) # Target

    """ Now to fill up the Numpy Arrays """
    for i, sentence in enumerate(sentences):
        for t, character in enumerate(sentence):
            """ Sentence number i at position t and character number at position char_to_index to 1 or True"""
            x[i, t, char_to_index[character]] = 1
        y[i, char_to_index[next_characters[i]]] = 1 # Same for target

    """ Model Creation """
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters)))) # LSTM(neurons, input_shape)
    model.add(Dense(len(characters))) # No. of neurons = Length of characters
    model.add(Activation('softmax')) # Scales output so its values adds up to 1, probability of next character

    """ Model Compilation """
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

    """ Model Training """
    model.fit(x, y, batch_size=256, epochs=20)

    """ Model Saving """
    model.save('poetic_text_generator.keras')

# if __name__ == "__main__":
#     main()

model = tf.keras.models.load_model('poetic_text_generator.keras')

# Helper Function from Keras Tutorial
""" Based on softmax probability, it will choose one character but it also depends on temperature, higher temprature means riskier choices """
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    """ Randomly assigns a starting character at an index, and takes 40 characters at a time, -1 because indices """
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence # Append sentence to generated text 
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
        
        predictions = model.predict(x, verbose = 0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated
'''
