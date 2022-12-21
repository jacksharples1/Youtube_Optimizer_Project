from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def initialize_model_tabular(shape=(1,)):
        input_num = layers.Input(shape=shape)

        x = layers.Dense(64, activation="relu")(input_num)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        output_num = layers.Dense(32, activation="relu")(x)

        model_num = models.Model(inputs=input_num, outputs=output_num)

        return model_num

def train_model_tabular(model,
                        X_train,
                        y_train,
                        epochs = 1000,
                        batch_size=32,
                        patience=2,
                        learning_rate_tabular= 0.001):
    es = EarlyStopping(patience=patience)

    model.compile(loss = "mae", optimizer=Adam(learning_rate=learning_rate_tabular), metrics=['mae'])
    history = model.fit(X_train, y_train,
            validation_split=0.3,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es]
            )

    return model, history
