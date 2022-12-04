import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple


print(Fore.BLUE + "\nLoading tensorflow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers


end = time.perf_counter()
print(f"\n✅ tensorflow loaded ({round(end - start, 2)} secs)")


def preprocess_images(X_train, X_test):
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    return X_train, X_test


def initialize_model_images(learning_rate) -> Model:
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
            AveragePooling2D(pool_size = (3,3)),
            Flatten(),
            Dense(50,activation = 'relu'),
            Dense(1,activation = 'linear')))

        optimizer = Adam(learning_rate=learning_rate)

        model.compile(loss="mae", optimizer=optimizer,
                    metrics=["mse"])

        return model

    model_images = complete_model()

    print("\n✅ Images model compiled")
    return model_images


def train_model_images(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                batch_size=16,
                patience=10,
                epochs=50,
                validation_split=0.2,) -> Tuple[Model, dict]:
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

    print(f"\n✅ Images model trained ({len(X_train)} rows)")
    return model, history



# NON-TRANSFER LEARNING MODEL
def simple_images(X_train,y_train):
    es = EarlyStopping(patience=30,restore_best_weights=True,)
    model_pipe = models.Sequential([
        layers.experimental.preprocessing.Rescaling(scale=1./255., input_shape=(180, 320, 3)),
        layers.Conv2D(16, (3,3), padding='same', activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(32, (2,2), padding='same', activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(64, (2,2), padding='same', activation="relu"),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(25, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])

    model_pipe.compile(loss='mse',
                optimizer='adam',
                metrics=['mae'])

    history = model_pipe.fit(X_train, y_train,
            epochs=1,  # Use early stopping in practice
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            callbacks=[es]
            )

    return model_pipe, history

# RESNET MODEL
def complex_images(X_train,y_train):
    from tensorflow.keras.applications.resnet import ResNet101

    def load_model():

        model = ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(180, 320, 3),
        pooling=None,
        # classes=1000,
        # classifier_activation='softmax',
        )

        return model


    def set_nontrainable_layers(model):

        model.trainable = False

        return model


    from tensorflow.keras import layers, models

    def add_last_layers(model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        base_model = model
        base_model = set_nontrainable_layers(base_model)
        flattening_layer = layers.Flatten()
        dense_layer1 = layers.Dense(500, activation='relu')
        drop1 = layers.Dropout(rate=0.3)
        prediction_layer = layers.Dense(1, activation='linear')

        model = models.Sequential([
        base_model,
        flattening_layer,
        dense_layer1,
        drop1,
        prediction_layer
        ])


        return model

    from tensorflow.keras import optimizers
    def build_model():
        opt = optimizers.Adam(learning_rate=0.01)
        base_model = load_model()
        model = add_last_layers(base_model)

        model.compile(loss='mse',
                    optimizer=opt,
                    metrics=['mae'])

        return model

    from tensorflow.keras.applications.resnet import preprocess_input

    X_train = preprocess_input(X_train)


    es = EarlyStopping(patience=10,restore_best_weights=True)

    model = build_model()

    history = model.fit(X_train, y_train,
            batch_size=16,
            epochs=2,
            validation_split=0.2,
            callbacks=[es])

    return model, history
