import numpy as np
import pandas as pd

# from tests.test_base import write_result

from youtube.ml_logic.images import initialize_model_images, train_model_images
from youtube.ml_logic.nlpcnn import initialize_model_nlp_cnn, train_model_nlp_cnn
from youtube.ml_logic.concat import initialize_model_concat, train_model_concat
from youtube.ml_logic.preprocessor import preprocess_features
from youtube.ml_logic.registry import save_model, load_model

from youtube.ml_logic.params import (
    DATASET,
    LOCAL_DATA_PATH)

from youtube.ml_logic.utils import simple_time_and_memory_tracker

from youtube.ml_logic.tokenizer import tokenizer

# def get_dataframe():
#     df = pd.read_csv(f"../../raw_data/{DATASET}")

#     return df

def preprocess(source_type='train'):
    """
    Preprocess the dataset iteratively by loading data in chunks that fit in memory,
    processing each chunk, appending each of them to a final dataset, and saving
    the final preprocessed dataset as a CSV file
    """

    print("\n⭐️ Use case: preprocess")

    df, images = preprocess_features()

    y = df.views
    X_nlp = df.title
    X_images = np.array(images)


    print("✅ data processed saved entirely")
    return X_images, X_nlp, y



def train_test_split(X_images, X_nlp, y):
    from sklearn.model_selection import train_test_split
    X_train_images, X_test_images,X_train_nlp,X_test_nlp, y_train, y_test = train_test_split(X_images,
                                                                                             X_nlp,
                                                                                             y,
                                                                                             test_size=0.2,
                                                                                             random_state=0)
    return X_train_images, X_test_images,X_train_nlp,X_test_nlp, y_train, y_test



@simple_time_and_memory_tracker
def preprocess_and_train():
    """
    Load historical data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    print("\n⭐️ Use case: preprocess and train basic")

    X_images, X_nlp, y = preprocess_features()

    X_train_images, X_test_images,X_train_nlp,X_test_nlp, y_train, y_test = train_test_split(X_images, X_nlp, y)

    # IMAGES

    model_images = initialize_model_images()

    batch_size = 16
    patience = 10

    model_images, history_images = train_model_images(model_images,
                                               X_train_images,
                                               y_train,
                                               batch_size=batch_size,
                                               patience=patience)


    # # Compute the validation metric (min val mae of the holdout set)
    metrics_images = dict(mae=np.min(history_images.history['val_loss']))
    print(metrics_images)



    # NLP


    X_train_pad, X_test_pad, input_length, vocab_size = tokenizer(X_train_nlp, X_test_nlp)



    model_nlp = initialize_model_nlp_cnn(input_length,vocab_size)

    batch_size = 16
    patience = 10


    model_nlp, history_nlp = train_model_nlp_cnn(model_nlp,
                                               X_train_pad,
                                               y_train,
                                               batch_size=batch_size,
                                               patience=patience)

    metrics_nlp = dict(mae=np.min(history_nlp.history['val_loss']))
    print(metrics_nlp)

    # Concat Model


    # Define Inputs and Outputs

    model_concat = initialize_model_concat(model_nlp, model_images)

    batch_size = 32
    patience = 10


    # train_model_concat(model_concat)

    model_concat, history_concat = train_model_concat(model_concat,
                                                      X_train_pad,
                                                      X_train_images,
                                                      y_train,
                                                      batch_size=batch_size,
                                                      patience=patience)

    metrics_concat = dict(mae=np.min(history_concat.history['val_loss']))
    print(metrics_concat)

    # # Save trained model
    # params = dict(
    #     # learning_rate=learning_rate,
    #     batch_size=batch_size,
    #     patience=patience
    # )
    # save_model(model, params=params, metrics=metrics)

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    # write_result(name="test_preprocess_and_train", subdir="train_at_scale", metrics=metrics)

    print("✅ preprocess_and_train() done")


@simple_time_and_memory_tracker
def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]
        ))

    model = load_model()

    # Preprocess the new data
    X_pred_processed = preprocess_features(X_pred)


    # Make a prediction
    y_pred = model.predict(X_pred_processed)

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    write_result(name="test_pred", subdir="train_at_scale", y_pred=y_pred)
    print("✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred



if __name__ == '__main__':
    try:
        # X_images, X_nlp, y = preprocess()
        # X_train_images, X_test_images,X_train_nlp,X_test_nlp, y_train, y_test = train_test_split(X_images, X_nlp, y)
        # preprocess(source_type='val')
        preprocess_and_train()
        # pred()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
