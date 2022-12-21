import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from youtube.ml_logic.params import TIMESTAMP
import pickle
import time

def tokenizer(X_train_nlp, X_test_nlp, evaluate):

    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(X_train_nlp)

    X_train_token = tokenizer.texts_to_sequences(X_train_nlp)

    X_train_pad = pad_sequences(X_train_token, padding='post', dtype='float32')

    input_length=np.array([len(i) for i in X_train_pad]).max()
    vocab_size = len(tokenizer.word_index)

    if evaluate:
        X_test_token = tokenizer.texts_to_sequences(X_test_nlp)
        X_test_pad = pad_sequences(X_test_token, padding='post', maxlen=input_length,  dtype='float32')

    else:
        X_test_pad = None

    # Set timestamp for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Saving tokenizer and input_length
    print('Pickling tokenizer and input_length...')

    with open(f'nlp_pickles/tokenizer_{timestamp}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'nlp_pickles/input_length_{timestamp}.pickle', 'wb') as handle:
        pickle.dump(input_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Pickled with timestamp: {timestamp}')

    return X_train_pad, X_test_pad, input_length, vocab_size, timestamp

def tokenizer_pred(X):
    
    # Loading tokenizer and input length
    with open(f'nlp_pickles/tokenizer_{TIMESTAMP}.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'nlp_pickles/input_length_{TIMESTAMP}.pickle', 'rb') as handle:
        input_length = pickle.load(handle)

    X_token = tokenizer.texts_to_sequences(X)
    return pad_sequences(X_token, padding='post', maxlen=input_length,  dtype='float32')
