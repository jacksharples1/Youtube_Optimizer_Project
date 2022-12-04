import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

from youtube.ml_logic.images import initialize_model_images, train_model_images, simple_images, complex_images
from youtube.ml_logic.nlpcnn import initialize_model_nlp_cnn, train_model_nlp_cnn
from youtube.ml_logic.nlprnn2 import initialize_model_nlp_rnn2, train_model_nlp_rnn2
from youtube.ml_logic.concat import initialize_model_concat, train_model_concat
from youtube.ml_logic.preprocessor import preprocess_features
from youtube.ml_logic.registry import save_model, load_model
# from youtube.ml_logic.utils import plot_loss_mse, simple_time_and_memory_tracker
from youtube.ml_logic.params import DATASET
from youtube.ml_logic.tokenizer import tokenizer

# @simple_time_and_memory_tracker
def preprocess():
    """
    Preprocess the dataset and saving the final preprocessed dataset as CSV files and a numpy array.
    """

    print("\n⭐️ Use case: preprocess")

    df, images = preprocess_features()

    # Set Xs and y
    X_images = np.array(images)
    X_nlp = df.title
    y = df.views

    # Save Xs and y
    np.save(f'./raw_data/{DATASET}_x_images.npy',X_images)
    X_nlp.to_csv(f'./raw_data/{DATASET}_x_nlp.csv',index=False)
    y.to_csv(f'./raw_data/{DATASET}_y.csv',index=False)

    print("✅ Data processed and saved")

    return None

# @simple_time_and_memory_tracker
def train(evaluate=False):
    """
    Load historical data in memory, clean and preprocess it, train a Keras model on it,
    save the model, and finally compute & save a performance metric
    on a validation set holdout at the `model.fit()` level
    """

    if evaluate:
        print("\n⭐️ Use case: train and evaluate")
        context = 'Train and evaluate'
    else:
        print("\n⭐️ Use case: train on all data")
        context = 'Train on all'

    # Load data
    X_images = np.load(f'./raw_data/{DATASET}_x_images.npy')
    X_nlp = pd.read_csv(f'./raw_data/{DATASET}_x_nlp.csv',index_col=False)['title']
    y = pd.read_csv(f'./raw_data/{DATASET}_y.csv',index_col=False)['views']

    # Train test split if evaluating
    if evaluate:
        X_train_images, X_test_images, X_train_nlp, X_test_nlp, y_train, y_test = train_test_split(X_images,X_nlp,y,test_size=0.3)
        print(f"After train test split, {len(X_train_images)} items used to train model")

    else:
        X_train_images,X_train_nlp, y_train = X_images, X_nlp, y
        X_test_nlp = None
        print(f"{len(X_train_images)} items used to train model")


    # Tokenize, save tokenizer, save input_length and set timestamp
    X_train_pad, X_test_pad, input_length, vocab_size, timestamp = tokenizer(X_train_nlp, X_test_nlp, evaluate)


    # Use to save examples for testing predictions
    if evaluate:
        np.save(f'./raw_data/single_nlp_pred.npy',X_test_pad[0:1])
        np.save(f'./raw_data/single_images_pred.npy',X_test_images[0:1])


    ## 1) #################### IMAGES ###################

    # Images params:
    batch_size_images = 16
    patience_images = 3
    # epochs_images = 10
    epochs_images = 1
    learning_rate_images = 0.1

    # Initialize and train images model
    model_images = initialize_model_images(learning_rate_images)
    model_images, history_images = train_model_images(model_images,
                                               X_train_images,
                                               y_train,
                                               epochs=epochs_images,
                                               batch_size=batch_size_images,
                                               patience=patience_images)

    # Compute the validation metric (min val mae of the holdout set)
    metrics_images = dict(mae=np.min(history_images.history['val_loss']))

    ## 2) #################### NLP ###################

    # NLP params:
    batch_size_nlp = 128
    patience_nlp = 5
    # epochs_nlp = 1000
    epochs_nlp = 1
    learning_rate_nlp = 0.001

    # Initialize and train NLP model
    model_nlp = initialize_model_nlp_rnn2(input_length,vocab_size, learning_rate_nlp)
    model_nlp, history_nlp = train_model_nlp_rnn2(model_nlp,
                                                  X_train_pad,
                                                  y_train,
                                                  batch_size=batch_size_nlp,
                                                  epochs = epochs_nlp,
                                                  patience=patience_nlp)

    # Compute the validation metric (min val mae of the holdout set)
    metrics_nlp = dict(mae=np.min(history_nlp.history['val_loss']))


    ## 3) #################### CONCAT ###################

    # Concat params:
    batch_size_concat = 32
    patience_concat = 10
    # epochs_concat = 100
    epochs_concat = 1
    learning_rate_concat = 0.001

    # Initialize and train Concat model
    model_concat = initialize_model_concat(model_nlp, model_images, learning_rate_concat)
    model_concat, history_concat = train_model_concat(model_concat,
                                                      X_train_pad,
                                                      X_train_images,
                                                      y_train,
                                                      epochs = epochs_concat,
                                                      batch_size=batch_size_concat,
                                                      patience=patience_concat)

    # plot_loss_mse(history_concat)
    metrics_concat = dict(mae=np.min(history_concat.history['val_mae']))

    if evaluate:
        print('Images metrics:')
        print(metrics_images['mae'])
        print('\nNLP metrics:')
        print(metrics_nlp['mae'])
        print('\nConcatinated metrics:')
        baseline = (abs(y_test-y_train.mean())).mean()
        result = model_concat.evaluate(x = [X_test_pad, X_test_images], y= y_test, verbose=1)
        print(f'Mean absolute error : {result[1]:.4f}')
        print(f'Baseline score is {baseline}')
        print(f'Improvement on baseline  {round((baseline-result[1])*100/baseline,2)}%')
        print("\n✅ train and evaluate done")
    else:
        print('Images metrics:')
        print(metrics_images)
        print('\nNLP metrics:')
        print(metrics_nlp)
        print('\nConcatinated metrics:')
        print(metrics_concat)
        # baseline = (abs(y_train-y_train.mean())).mean()
        score = metrics_concat['mae']
        print(f'Mean absolute error : {score:.4f}')
        # print(f'Val mae improvement on baseline {((baseline - score)/baseline)*100}%')
        print("\n✅ train done")

    # Save trained model
    params = dict(
        # Model parameters
            #Images
        batch_size_images = batch_size_images,
        patience_images = patience_images,
        epochs_images = epochs_images,
        learning_rate_images = learning_rate_images,
            #NLP
        batch_size_nlp = batch_size_nlp,
        patience_nlp = patience_nlp,
        epochs_nlp = epochs_nlp,
        learning_rate_nlp = learning_rate_nlp,
            #Concat
        batch_size_concat = batch_size_concat,
        patience_concat = patience_concat,
        epochs_concat = epochs_concat,
        learning_rate_concat = learning_rate_concat,

        # Package behavior
        context=context,

        # Data source
        dataset=DATASET,
        row_count=len(X_train_images),
        dataset_timestamp=timestamp,
        )

    if evaluate:
        metrics = result[1]
    else:
        metrics = score

    save_model(model=model_concat, params=params, metrics=dict(mae=metrics), timestamp=timestamp)


def pred(X_pred=None, model_name=None):

    if X_pred is None:
        X_pred_nlp = np.load(f'./raw_data/single_nlp_pred.npy')
        X_pred_images = np.load(f'./raw_data/single_images_pred.npy')
        X_pred = [X_pred_nlp,X_pred_images]

    model = load_model(model_name=False)

    # # Preprocess the new data
    # X_pred_processed = preprocess_features(X_pred)

    # Make a prediction
    y_pred = model.predict(X_pred)

    print("✅ prediction done: ", y_pred, y_pred.shape)

    return y_pred


if __name__ == '__main__':
    try:
        preprocess()
        train(evaluate=True)

    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
