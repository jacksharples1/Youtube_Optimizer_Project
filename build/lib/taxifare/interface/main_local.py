import numpy as np
import pandas as pd
import os

from tests.test_base import write_result

from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.model import initialize_model, compile_model, train_model
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, load_model

from taxifare.ml_logic.params import (
    CHUNK_SIZE,
    DTYPES_RAW_OPTIMIZED_HEADLESS,
    DTYPES_RAW_OPTIMIZED,
    DTYPES_PROCESSED_OPTIMIZED,
    COLUMN_NAMES_RAW,
    DATASET_SIZE,
    VALIDATION_DATASET_SIZE,
    LOCAL_DATA_PATH
)
from taxifare.ml_logic.utils import simple_time_and_memory_tracker

def preprocess(source_type='train'):
    """
    Preprocess the dataset iteratively by loading data in chunks that fit in memory,
    processing each chunk, appending each of them to a final dataset, and saving
    the final preprocessed dataset as a CSV file
    """

    print("\n⭐️ Use case: preprocess")

    # Local saving paths given to you (do not overwrite these data_path variables)
    source_name = f"{source_type}_{DATASET_SIZE}.csv"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}.csv"

    data_raw_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "raw", source_name))
    data_processed_path = os.path.abspath(os.path.join(LOCAL_DATA_PATH, "processed", destination_name))

    # Iterate over the dataset, in chunks
    chunk_id = 0

    FIRST = True
    # Let's loop until we reach the end of the dataset, then `break` out
    while True:
        print()
        print(f"processing chunk n°{chunk_id}...")
        skip_no = chunk_id*CHUNK_SIZE
        try:
            # Load the chunk numbered `chunk_id` of size `CHUNK_SIZE` into memory
            # 🎯 Hint: check out pd.read_csv(skiprows=..., nrows=..., headers=...)
            # We advise you to always load data with `header=None`, and add back column names using COLUMN_NAMES_RAW

            data = pd.read_csv(data_raw_path,dtype=DTYPES_RAW_OPTIMIZED,skiprows=skip_no+1,nrows=CHUNK_SIZE,header=None)


        except pd.errors.EmptyDataError:
            # 🎯 Hint: what would you do when you reach the end of the CSV?
            break


        # Clean chunk; pay attention, sometimes it can result in 0 rows remaining!
        if not data.empty:
            data.columns = COLUMN_NAMES_RAW
            data_cleaned = clean_data(data)

            # Create X_chunk, y_chunk
            X_chunk = data_cleaned.drop("fare_amount", axis=1)
            y_chunk = data_cleaned[["fare_amount"]]

            # Create X_processed_chunk and concatenate (X_processed_chunk, y_chunk) into data_processed_chunk
            X_processed_chunk = preprocess_features(X_chunk)
            # print(type(X_processed_chunk))
            # X_processed_chunk = pd.DataFrame(X_processed_chunk)

            print('SHAPES')
            print(X_processed_chunk.shape)
            print(y_chunk.shape)
            data_processed_chunk = pd.DataFrame(np.concatenate([X_processed_chunk,y_chunk],axis=1))
            print(data_processed_chunk.shape)
            # Save and append the chunk of the preprocessed dataset to a local CSV file
            # Keep headers on the first chunk: for convention, we'll always save CSVs with headers in this challenge
            # 🎯 Hints: check out pd.to_csv(mode=...)

            if FIRST:
                data_processed_chunk.to_csv(data_processed_path, mode='w', header=True, index=False)
                FIRST = False
            else:
                
                data_processed_chunk.to_csv(data_processed_path, mode='a', header=False, index=False)


        chunk_id += 1


    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    data_processed = pd.read_csv(data_processed_path, header=None, skiprows=1, dtype=DTYPES_PROCESSED_OPTIMIZED).to_numpy()
    write_result(name="test_preprocess", subdir="train_at_scale", data_processed_head=data_processed[0:10])


    print("✅ data processed saved entirely")

@simple_time_and_memory_tracker
def preprocess_and_train():
    """
    Load historical data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    print("\n⭐️ Use case: preprocess and train basic")


    # Retrieve raw data
    data_raw_path = os.path.join(LOCAL_DATA_PATH, "raw", f"train_{DATASET_SIZE}.csv")
    data = pd.read_csv(data_raw_path, dtype=DTYPES_RAW_OPTIMIZED)

    # Clean data using ml_logic.data.clean_data
    data_cleaned = clean_data(data)

    # Create X, y
    X = data_cleaned.drop("fare_amount", axis=1)
    y = data_cleaned[["fare_amount"]]

    # Preprocess X using `preprocessor.py`
    X_processed = preprocess_features(X)

    # Train model on X_processed and y, using `model.py`
    model = initialize_model(X_processed)
    learning_rate = 0.001
    compile_model(model,learning_rate=learning_rate)

    batch_size = 256
    patience = 2
    model, history = train_model(model,
                X_processed,
                y,
                batch_size=batch_size,
                patience=patience,
                validation_split=0.3,
                validation_data=None)


    # Compute the validation metric (min val mae of the holdout set)
    metrics = dict(mae=np.min(history.history['val_mae']))

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_model(model, params=params, metrics=metrics)

    # 🧪 Write outputs so that they can be tested by make test_train_at_scale (do not remove)
    write_result(name="test_preprocess_and_train", subdir="train_at_scale", metrics=metrics)

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
        preprocess()
        preprocess(source_type='val')
        # preprocess_and_train()
        # pred()
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
