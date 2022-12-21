from datetime import datetime
# $WIPE_BEGIN
import pickle
import numpy as np
from youtube.ml_logic.processnlp import preprocessing
from youtube.ml_logic.registry import load_model
from youtube.ml_logic.params import TIMESTAMP, LOG
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, models

# $WIPE_END
import json
from fastapi import FastAPI, Request
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

@app.post("/test_predict")
async def test_predict(input: Request):

    data = await input.json()
    data = eval(data)
    string = data['text']
    image = np.array(data['image'])

    # Loading tokenizer and input length
    with open(f'nlp_pickles/tokenizer_{TIMESTAMP}.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'nlp_pickles/input_length_{TIMESTAMP}.pickle', 'rb') as handle:
        input_length = pickle.load(handle)

    #TODO rewrite this section
    # Process real input
    string_preprocessed = []
    for title in string:
        title_preprocessed = preprocessing(title)
        string_preprocessed.append(title_preprocessed)

    # Tokenizer
    X_test_token = tokenizer.texts_to_sequences(string_preprocessed)
    X_test_pad = pad_sequences(X_test_token, padding='post', maxlen=input_length,  dtype='float32')


    images_list = []
    titles_list = []
    indices = []
    index = 0
    for i in X_test_pad:
        images_list.extend(image)
        for n in range(len(image)):
            titles_list.extend([i])
            indices.append((index,n))
        index +=1

    X_pred_comb = [np.array(titles_list),np.array(images_list)]

    model = app.state.model
    try:
        y_pred_comb = model.predict(X_pred_comb)
        if LOG:
            y_pred_comb = np.exp(y_pred_comb) -1
        print('Y_pred_comb')
        print(y_pred_comb)
        print(indices)

    except Exception as e:
        print(e)

    result_dict = {'prediction':[y_pred_comb],'index':[indices]}
    print(y_pred_comb[0][0])
    print(indices[0])
    result_dict_single = {'prediction':y_pred_comb[0][0],'index':indices[0]}
    print(result_dict)


    return dict(prediction = f"{y_pred_comb}",
                index=f"{indices}",
                )

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(temporary="Homepage")
    # $CHA_END
