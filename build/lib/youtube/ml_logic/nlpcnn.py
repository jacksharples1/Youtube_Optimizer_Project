from tensorflow.keras import Sequential,layers, Model
from typing import Tuple
import numpy as np

def initialize_model_nlp_cnn(input_length,vocab_size, embedding_size=100):
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

    model.compile(
        loss='mae',
        optimizer='adam',
        metrics='mae'
    )

    return model

    # reg = regularizers.l1_l2(l2=0.01)

    # model = Sequential()
    # model.add(layers.BatchNormalization(input_shape=X.shape[1:]))
    # model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg, input_shape=X.shape[1:]))
    # model.add(layers.BatchNormalization())

    # model.add(layers.Dense(50, activation="relu", kernel_regularizer=reg))
    # model.add(layers.BatchNormalization())

    # model.add(layers.Dense(10, activation="relu"))
    # model.add(layers.BatchNormalization(momentum=0.99)) # use momentum=0 for to only use statistic of the last seen
    # # minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.

    # model.add(layers.Dense(1, activation="linear"))

    # print("\n✅ model initialized")

    # return model

# @simple_time_and_memory_tracker
# def compile_model(model: Model, learning_rate: float) -> Model:
#     """
#     Compile the Neural Network
#     """
#     print(f'model type is {type(model)}')
#     optimizer = optimizers.Adam(learning_rate=learning_rate)
#     model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

#     print("\n✅ model compiled")
#     return model


def train_model_nlp_cnn(model: Model,X_train: np.ndarray,y_train: np.ndarray,batch_size=16,patience=10,epochs=50,validation_split=0.2,validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    es = EarlyStopping(patience = patience,
                       restore_best_weights = True)

    history = model.fit(X_train,
                        y_train,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_split=validation_split,
                        callbacks = [es])

    return model, history
#     es = EarlyStopping(
#     monitor="val_loss",
#     patience=patience,
#     restore_best_weights=True,
#     verbose=0
# )

#     history = model.fit(
#         X,
#         y,
#         validation_split=validation_split,
#         epochs=100,
#         batch_size=batch_size,
#         callbacks=[es],
#         verbose=1)

#     print(f"\n✅ model trained ({len(X)} rows)")

#     return model, history
