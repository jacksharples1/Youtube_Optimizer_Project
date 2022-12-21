import numpy as np
import pandas as pd
from math import log, exp
import pickle
from sklearn.model_selection import train_test_split
from youtube.ml_logic.processnlp import preprocessing
from youtube.ml_logic.images import initialize_model_images, train_model_images, simple_images, complex_images
from youtube.ml_logic.nlp import initialize_model_nlp_rnn2, train_model_nlp_rnn2
from youtube.ml_logic.tabular import initialize_model_tabular, train_model_tabular
from youtube.ml_logic.concat import initialize_model_concat, train_model_concat
from youtube.ml_logic.preprocessor import preprocess_features
from youtube.ml_logic.features import add_features
from youtube.ml_logic.registry import save_model, load_model
from youtube.ml_logic.params import DATASET, TIMESTAMP, TABULAR, LOG
from youtube.ml_logic.tokenizer import tokenizer, tokenizer_pred
from youtube.ml_logic.utils import plot_loss_mse, simple_time_and_memory_tracker


# @simple_time_and_memory_tracker
def preprocess():
    """
    Preprocess the dataset and saving the final preprocessed dataset as CSV files and a numpy array.
    """

    print("\n⭐️ Use case: preprocess")

    df, images = preprocess_features()

    df = add_features(df)

    # Set Xs and y
    X_images = np.array(images)
    X_nlp = df.title
    # Add any columns to be used for tabular data
    X_tabular= df[['channel_subscriberCount']]

    y = df.views

    # Save Xs and y
    np.save(f'./raw_data/{DATASET}_x_images.npy',X_images)
    X_nlp.to_csv(f'./raw_data/{DATASET}_x_nlp.csv',index=False)
    X_tabular.to_csv(f'./raw_data/{DATASET}_x_tabular.csv',index=False)
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
    X_tabular = pd.read_csv(f'./raw_data/{DATASET}_x_tabular.csv',index_col=False)
    y = pd.read_csv(f'./raw_data/{DATASET}_y.csv',index_col=False)['views']

    if LOG:
        y = y.apply(lambda x: log(x+1))

    # Train test split if evaluating
    if evaluate:
        X_train_images, X_test_images, X_train_nlp, X_test_nlp, X_train_tabular, X_test_tabular, y_train, y_test = train_test_split(X_images,X_nlp,X_tabular,y,test_size=0.3)
        print(f"After train test split, {len(X_train_images)} items used to train model")

    else:
        X_train_images,X_train_nlp, X_train_tabular, y_train = X_images, X_nlp, X_tabular, y
        X_test_nlp = None

        print(f"{len(X_train_images)} items used to train model")


    # Tokenize, save tokenizer, save input_length and set timestamp
    X_train_pad, X_test_pad, input_length, vocab_size, timestamp = tokenizer(X_train_nlp, X_test_nlp, evaluate)


    ## 1) #################### IMAGES ###################

    # Images params:
    batch_size_images = 16
    patience_images = 3
    # epochs_images = 10
    epochs_images = 20
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
    plot_loss_mse(history_images)

    ## 2) #################### NLP ###################

    # NLP params:
    batch_size_nlp = 128
    patience_nlp = 5
    # epochs_nlp = 1000
    epochs_nlp = 1000
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
    plot_loss_mse(history_nlp)

    ## 3) #################### TABULAR ###################

    if TABULAR:

        # tabular params:
        batch_size_tabular = 32
        patience_tabular = 10
        # epochs_tabular = 100
        epochs_tabular = 1000
        learning_rate_tabular = 0.001
        breakpoint()
        shape = (X_train_tabular.shape[1],)
        model_tabular = initialize_model_tabular(shape=shape)

        model_tabular, history_tabular = train_model_tabular(model_tabular,
                                                            X_train_tabular,
                                                            y_train,
                                                            epochs = epochs_tabular,
                                                            batch_size=batch_size_tabular,
                                                            patience=patience_tabular,
                                                            learning_rate_tabular= learning_rate_tabular)

        metrics_tabular = dict(mae=np.min(history_tabular.history['val_mae']))

    ## 4) #################### CONCAT ###################

    # Concat params:
    batch_size_concat = 32
    patience_concat = 6
    # epochs_concat = 100
    epochs_concat = 100
    learning_rate_concat = 0.001

    # Initialize and train Concat model
    if not TABULAR:
        model_tabular = None
        X_train_tabular = None

    model_concat = initialize_model_concat(model_nlp, model_images, model_tabular, learning_rate_concat)
    model_concat.summary()
    print("Model initialized")


    model_concat, history_concat = train_model_concat(model_concat,
                                                      X_train_pad,
                                                      X_train_images,
                                                      X_train_tabular,
                                                      y_train,
                                                      tabular = TABULAR,
                                                      epochs = epochs_concat,
                                                      batch_size=batch_size_concat,
                                                      patience=patience_concat)

    plot_loss_mse(history_concat)
    metrics_concat = dict(mae=np.min(history_concat.history['val_mae']))

    print('Images metrics:')
    print(metrics_images['mae'])
    print('\nNLP metrics:')
    print(metrics_nlp['mae'])
    if TABULAR:
        print('\nTabular metrics:')
        print(metrics_tabular)
    print('\nConcatinated metrics:')
    if evaluate:
        if LOG:
            y_test_unlog  = y_test.apply(lambda x: exp(x)-1)
            y_train_unlog  = y_train.apply(lambda x: exp(x)-1)
            baseline = (abs(y_test_unlog-y_train_unlog.mean())).mean()
            if TABULAR:
                X_pred = [X_test_pad,X_test_images,X_test_tabular]
            else:
                X_pred = [X_test_pad, X_test_images]
            y_predict = model_concat.predict(X_pred)
            y_predict_unlog = np.exp(y_predict)-1
            score = (abs(y_test_unlog-y_predict_unlog.mean())).mean()
        else:
            baseline = (abs(y_test-y_train.mean())).mean()
            if TABULAR:
                result = model_concat.evaluate(x = [X_test_pad, X_test_images, X_test_tabular], y= y_test, verbose=1)
            else:
                result = model_concat.evaluate(x = [X_test_pad, X_test_images], y= y_test, verbose=1)
            score = result[1]
        print(f'Mean absolute error : {score:.4f}')
        percentage_improvement = round((baseline-score)*100/baseline,2)
        print(f'Baseline score is {baseline}')
        print(f'Improvement on baseline {percentage_improvement}%')
        print("\n✅ train and evaluate done")
    else:
        print(metrics_concat)
        score = metrics_concat['mae']
        print(f'Mean absolute error : {score:.4f}')
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
        tabular= TABULAR,
        log=LOG,
        )

    if not evaluate:
        percentage_improvement= np.nan

    save_model(model=model_concat, params=params, metrics=dict(percentage_improvement = percentage_improvement,mae=score), timestamp=timestamp)

def pred(X_pred=None, model_name=None):

    if X_pred is None:
        X_pred_nlp = pd.DataFrame([['Learn to code like a wizard: Harry Potter style'], ['Learn to code like a wizard: Harry Potter style'], ['Learn to code like a wizard: Harry Potter style']], columns=['title'])
        X_pred_images = np.load(f'./raw_data/test_images.npy')
        X_pred_sub = pd.DataFrame([[100_000], [100_000], [100_000]], columns=['channel_subscriberCount'])

    model = load_model(model_name=True)

    X_test_pad = tokenizer_pred(X_pred_nlp['title'].apply(lambda x: preprocessing(x)))

    if TABULAR:
        X_pred = [X_test_pad,X_pred_images,X_pred_sub]

    else:
        X_pred = [X_test_pad,X_pred_images]

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
