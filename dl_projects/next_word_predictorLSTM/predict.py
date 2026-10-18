# from models import next_word_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from keras.models import load_model
import os, pickle
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

curr_path = os.path.dirname(os.path.abspath(__file__))

text = "serious"
text = text.lower()
with open(f"{curr_path}/models/tokenizer.pkl", "rb") as f:
    toker = pickle.load(f)


# tokenise
token_text = toker.texts_to_sequences([text])[0]
lstm = load_model(f"{curr_path}/models/next_word_model.h5")


if text in toker.word_index :
    for i in range(5):
        padded_token_text = pad_sequences([token_text], maxlen= lstm.input_shape[1], padding= 'pre')
        # print(padded_token_text)
        pos = np.argmax(lstm.predict(padded_token_text, verbose = 0)) 
        for word, index in toker.word_index.items():
            if index == pos:
                text = text + " " + word
                print(text)
                
else : 
    print(f'Your text "{text}" was not in the training data')
    
