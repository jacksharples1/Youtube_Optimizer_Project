
# from colorama import Fore, Style

# import time
# print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
# start = time.perf_counter()

from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import preprocess_input



# end = time.perf_counter()
# print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")

from typing import Tuple

import numpy as np
# from taxifare.ml_logic.utils import simple_time_and_memory_tracker

def preprocess_images(X_train, X_test):
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    return X_train, X_test


def initialize_model(X: np.ndarray) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    def base_model():
        base_model = Xception(weights="imagenet",input_shape = (180,320,3),include_top=False)
        base_model.trainable = False
        return base_model

    def complete_model():
        model = Sequential((
            base_model(),
            #GlobalAveragePooling2D(),
            AveragePooling2D(pool_size = (3,3)),
            Flatten(),
            Dense(50,activation = 'relu'),
            Dense(1,activation = 'linear')))

        opt = Adam(learning_rate=0.01,
                beta_1=0.9,
                beta_2=0.99)

        model.compile(loss="mae", optimizer='adam',
                    metrics=["mse"])

        return model

    model_images = complete_model()

    return model_images

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


def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                batch_size=16,
                patience=10,
                epochs=50,
                validation_split=0.2,
                validation_data=None) -> Tuple[Model, dict]:
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
