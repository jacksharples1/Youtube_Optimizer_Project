from tensorflow.keras import layers, Sequential, regularizers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Tuple

def initialize_model_nlp_rnn2(input_length,vocab_size, learning_rate):

    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1,
        input_length=input_length, # Max_sentence_length (optional, for model summary)
        output_dim=embedding_size,
        mask_zero=True, # Built-in masking layer
    ))

    model.add(layers.GRU(50,return_sequences=True,kernel_regularizer=regularizers.L2(l2=0.001)))
    model.add(layers.Dropout(0.4))

    model.add(layers.GRU(30,return_sequences=True,kernel_regularizer=regularizers.L2(l2=0.001)))
    model.add(layers.Dropout(0.2))

    model.add(layers.GRU(20,return_sequences=False))
    model.add(layers.Dropout(0.2))


    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))


    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))

    # model.add(layers.Dense(8, activation="relu" ))
    model.add(layers.Dense(1,activation = 'linear'))


    optimizer=Adam(learning_rate=learning_rate)

    model.compile(
    loss='mae',
    optimizer=optimizer,
    metrics='mse'
    )

    print("\n✅ NLP model compiled")
    return model


def train_model_nlp_rnn2(model,
                         X_train,
                         y_train,
                         batch_size=128,
                         validation_split=0.2,
                         epochs=1000,
                         patience=5):
    es=EarlyStopping(patience=patience, restore_best_weights=True)

    history=model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[es],
                    epochs=epochs)

    print(f"\n✅ NLP model trained ({len(X_train)} rows)")
    return model, history

def initialize_model_nlp_cnn(input_length,vocab_size, embedding_size=100, learning_rate=0.01):
    model=Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size+1, # 16 +1 for the 0 padding
        input_length=input_length, # Max_sentence_length (optional, for model summary)
        output_dim=embedding_size, # 100
        mask_zero=True, # Built-in masking layer :)
    ))



    model.add(layers.Conv1D(20, kernel_size=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='linear'))

    optimizer=Adam(learning_rate=learning_rate)

    model.compile(
        loss='mae',
        optimizer=optimizer,
        metrics='mae'
    )

    return model

def train_model_nlp_cnn(model: Model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        batch_size=16,
                        patience=10,
                        epochs=1000,
                        validation_split=0.2,
                        validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es=EarlyStopping(patience=patience, restore_best_weights=True)

    history=model.fit(X_train,
                      y_train,
                      batch_size=batch_size,
                      validation_split=validation_split,
                      callbacks=[es],
                      epochs=epochs)

    return model, history
