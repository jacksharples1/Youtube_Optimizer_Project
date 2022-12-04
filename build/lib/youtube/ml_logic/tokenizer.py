from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenizer(X_train_nlp, X_test_nlp):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(X_train_nlp)



    X_train_token = tokenizer.texts_to_sequences(X_train_nlp)
    X_test_token = tokenizer.texts_to_sequences(X_test_nlp)

    X_train_pad = pad_sequences(X_train_token, padding='post', dtype='float32')
    input_length=np.array([len(i) for i in X_train_pad]).max()
    X_test_pad = pad_sequences(X_test_token, padding='post', maxlen=input_length,  dtype='float32')

    vocab_size = len(tokenizer.word_index)


    return X_train_pad, X_test_pad, input_length, vocab_size
