from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential 

# here vocab_size is total number of unique words in y, seq_len is numbers of columns in padded y
def lstm_model(vocab_size, seq_len):
    lstm = Sequential()
    lstm.add(Embedding(vocab_size + 1 , 100, input_length=seq_len))
    lstm.add(LSTM(500))
    lstm.add(Dense(vocab_size + 1, activation='softmax'))

    lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(lstm.summary())
    return lstm

