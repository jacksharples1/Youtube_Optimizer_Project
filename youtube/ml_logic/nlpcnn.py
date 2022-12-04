from tensorflow.keras import Sequential,layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


from typing import Tuple
import numpy as np

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
