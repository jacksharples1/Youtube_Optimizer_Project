import mlflow
import glob
from youtube.ml_logic.params import TIMESTAMP
from colorama import Fore, Style
from tensorflow.keras import  models


def save_model(model,params,metrics, timestamp):

    timestamp = timestamp

    # Set mlflow env params
    # $CHA_BEGIN
    mlflow_tracking_uri = 'https://mlflow.lewagon.ai'
    mlflow_experiment = 'youtube_optimizer_experiment_jacksharples1'
    mlflow_model_name = 'youtube_optimizer_jacksharples1'
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

        # TODO: workout how to save models to mlflow
        # $CHA_BEGIN
        # if model is not None:
        #     print('Saving model to mlflow...')
        #     mlflow.keras.log_model(keras_model=model,
        #                             artifact_path="model",
        #                             keras_module="tensorflow.keras",
        #                             registered_model_name=mlflow_model_name)
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


    # if os.environ.get("MODEL_TARGET") == "mlflow":
    #     stage = "Production"

    #     print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

    #     # load model from mlflow
    #     model = None
    #     # $CHA_BEGIN
    #     mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    #     mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    #     model_uri = f"models:/{mlflow_model_name}/{stage}"
    #     print(f"- uri: {model_uri}")

    #     try:
    #         model = mlflow.keras.load_model(model_uri=model_uri)
    #         print("\n✅ model loaded from mlflow")
    #     except:
    #         print(f"\n❌ no model in stage {stage} on mlflow")
    #         return None

    #     if save_copy_locally:
    #         from pathlib import Path

    #         # Create the LOCAL_REGISTRY_PATH directory if it does exist
    #         Path(LOCAL_REGISTRY_PATH).mkdir(parents=True, exist_ok=True)
    #         timestamp = time.strftime("%Y%m%d-%H%M%S")
    #         model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
    #         model.save(model_path)
    #     # $CHA_END

    #     return model
