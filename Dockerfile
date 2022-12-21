# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
# FROM python:3.8.12-buster
# WORKDIR /prod
# COPY taxifare taxifare
# COPY requirements.txt requirements.txt
# COPY setup.py setup.py
# RUN pip install .
# CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT

####### 👇 OPTIMIZED SOLUTION (x86)👇  (May be too advanced for ML-Ops module but useful for the project weeks) #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0
# OR for apple silicon, use this base image instead
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

WORKDIR /prod
COPY youtube youtube

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements_prod.txt requirements.txt
COPY setup.py setup.py

COPY api_test.py api_test.py
COPY nltk_data /home/nltk_data
COPY models/model_whole_20221204-230057 models/model_whole_20221204-230057
COPY nlp_pickles/tokenizer_20221204-230057.pickle nlp_pickles/tokenizer_20221204-230057.pickle
COPY nlp_pickles/input_length_20221204-230057.pickle nlp_pickles/input_length_20221204-230057.pickle

RUN pip install -r requirements.txt

# # Copy .env with DATA_SOURCE=local and MODEL_TARGET=mlflow
# COPY .env .env

# Then, at run time, load the model locally from the container instead of querying the MLflow server, thanks to "MODEL_TARGET=local"
# This avoids to download the heavy model from the Internet every time an API request is performed
CMD uvicorn youtube.fast_api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END