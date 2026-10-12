import json 
import os, pickle
from utils import create_data, calculate_max_len, pad_data, get_x_y, toker
from model import lstm_model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle



curr_path = os.path.dirname(os.path.abspath(__file__))
with open(f"{curr_path}/data.json", 'r') as d:
    data = json.load(d)



prepared_data, unique_words = create_data(data['text'])
padded_seq_max_len = calculate_max_len(prepared_data)
# print(unique_words) # 826
# print(padded_seq_max_len) # 138


padded_data = pad_data(prepared_data ,padded_seq_max_len )
(x, y) = get_x_y(padded_data)

print(x.shape[1])
# print(x)
# print(y)


y_catogories = to_categorical(y, num_classes = unique_words + 1) 

lstm = lstm_model(unique_words , x.shape[1])
lstm.fit(x, y_catogories, epochs=100)

# save model for prediction 
lstm.save(f"{curr_path}/models/next_word_model.h5")

# save the tokenizer for later
with open(f"{curr_path}/models/tokenizer.pkl", "wb") as f:
    pickle.dump(toker, f)







