from datetime import datetime
# $WIPE_BEGIN
import pickle
import numpy as np
from youtube.ml_logic.processnlp import preprocessing
from youtube.ml_logic.registry import load_model
from youtube.ml_logic.params import TIMESTAMP
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, models

# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!


app.state.model = load_model(model_name=True)
# $WIPE_END


@app.get("/test_predict")
def test_predict(string):

    model = app.state.model

    # Loading tokenizer and input length
    with open(f'nlp_pickles/tokenizer_{TIMESTAMP}.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'nlp_pickles/input_length_{TIMESTAMP}.pickle', 'rb') as handle:
        input_length = pickle.load(handle)

    # Process real input
    string_preprocessed = preprocessing(string)


    # Tokenizer
    X_test_token = tokenizer.texts_to_sequences([string_preprocessed])
    X_test_pad = pad_sequences(X_test_token, padding='post', maxlen=input_length,  dtype='float32')



    # Preprepared testing inputs
    X_pred_nlp_test = np.load(f'./raw_data/single_nlp_pred.npy')
    X_pred_images_test = np.load(f'./raw_data/single_images_pred.npy')


    X_pred = [X_test_pad,X_pred_images_test]

    y_pred = model.predict(X_pred)

    return dict(guess = string,
                prediction=f'{int(y_pred[0][0],)} views',
                )

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(temporary="Homepage")
    # $CHA_END
