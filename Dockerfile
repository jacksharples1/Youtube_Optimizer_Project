
FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0

WORKDIR /prod
COPY youtube youtube

COPY requirements_prod.txt requirements.txt
COPY setup.py setup.py

COPY api_test.py api_test.py
COPY nltk_data /home/nltk_data
COPY models/model_whole_20221220-232504 models/model_whole_20221220-232504
COPY nlp_pickles/tokenizer_20221220-232504.pickle nlp_pickles/tokenizer_20221220-232504.pickle
COPY nlp_pickles/input_length_20221220-232504.pickle nlp_pickles/input_length_20221220-232504.pickle

RUN pip install -r requirements.txt

CMD uvicorn youtube.fast_api.fast:app --host 0.0.0.0 --port $PORT
