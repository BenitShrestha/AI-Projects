from main import clean_up_sentence, bag_of_words, predict_class, get_response

# Indicate the Chatbot is running
print('Chatbot is running!')

while (message != 'exit'):
    message = input("")
    ints = predict_class(message)
    if ints:
        res = get_response(ints, intents)
    else:
        res = "Sorry, I didn't get that. Please try again."
    print(res)
