import mlflow
import os
from youtube.ml_logic.params import TIMESTAMP
from colorama import Fore, Style
from tensorflow.keras import  models


def save_model(model,params,metrics, timestamp):

    timestamp = timestamp

    # Set mlflow env params
    # $CHA_BEGIN
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    mlflow_experiment = os.environ.get('MLFLOW_EXPERIMENT')
    # $CHA_END

    # configure mlflow
    # $CHA_BEGIN
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment)
    # $CHA_END

    with mlflow.start_run():

        # STEP 1: push parameters to mlflow
        # $CHA_BEGIN
        if params is not None:
            ('Saving params to mlflow...')
            mlflow.log_params(params)
            print("\n✅ Params saved to mlflow")
        # $CHA_END

        # STEP 2: push metrics to mlflow
        # $CHA_BEGIN
        if metrics is not None:
            print('Saving metrics to mlflow...')
            mlflow.log_metrics(metrics)
            print("\n✅ Metrics saved to mlflow")
        # $CHA_END

    if model is not None:
        print('Local save...')

        model.save(f'./models/model_whole_{timestamp}')
        print("\n✅ Model saved locally...")


def load_model(model_name):
    """
    load the latest saved model, return None if no model found
    """

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version

    if model_name:
        model_path = f"models/model_whole_{TIMESTAMP}"
        print(f"- path: {model_path}")

    else:
        results = glob.glob(f"models/*")
        if not results:
            print('No models found in directory')
            return None
        print(results)
        model_path = sorted(results)[-1]
        print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
