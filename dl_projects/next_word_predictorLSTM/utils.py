import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences    # pyright: ignore[reportMissingImports]


toker = Tokenizer()

def create_data(text_data):
    toker.fit_on_texts([text_data])
    input_sequences = []
    
    # storing each sentence in 'sentences' variable
    for sentences in text_data.split(','):
        # converting the sentence into vector
        sentence_vector = toker.texts_to_sequences([sentences])[0]
        # converting the vectorised sentence into input format for model
        for i in range(1, len(sentence_vector)):
            input_sequences.append(sentence_vector[ : i+1])
            
    return(input_sequences)
        
        
# calculate the length of all input data & return the highest one 
def calculate_max_len(input_sequences):
    single_max_len = max([len(x) for x in input_sequences])
    return single_max_len

# add Zero padding to input data 
def pad_data(input_sequences,single_max_len):
    padded_input = pad_sequences(input_sequences, maxlen = single_max_len, padding = 'pre')
    return padded_input