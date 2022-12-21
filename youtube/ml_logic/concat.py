from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model_concat(model_nlp, model_images, model_tabular, learning_rate=1e-4):

    input_nlp = model_nlp.input
    output_nlp = model_nlp.output

    input_images = model_images.input
    output_images = model_images.output

    if model_tabular:
        input_tabular = model_tabular.input
        output_tabular = model_tabular.output
        inputs = [input_nlp, input_images, input_tabular]
        combined = layers.concatenate([output_nlp, output_images, output_tabular])

    else:
        inputs = [input_nlp, input_images]
        combined = layers.concatenate([output_nlp, output_images])

    x = layers.Dense(10, activation="relu")(combined)

    outputs = layers.Dense(1, activation="linear")(x)

    model_combined = models.Model(inputs=inputs, outputs=outputs)

    model_combined.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

    print("\n✅ Concat model compiled")
    return model_combined


def train_model_concat(model,
                       X_train_pad,
                       X_train_images,
                       X_train_tabular,
                       y_train,
                       tabular=False,
                       batch_size=32,
                       patience=10,
                       epochs=1000,
                       validation_split=0.2):


    es = EarlyStopping(patience=patience)

    if tabular:
        x=[X_train_pad, X_train_images, X_train_tabular]
    else:
        x=[X_train_pad, X_train_images]

    history = model.fit(x=x,
                        y=y_train,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[es])


    print(f"\n✅ Concat model trained ({len(X_train_pad)} rows)")
    return model, history
