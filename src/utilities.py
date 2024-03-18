import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model
from sklearn.metrics import r2_score, mean_squared_error

from src.parameters import prediction_metric, max_layers


def split_data(
        dataset: pd.DataFrame,
        lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split dataset into values and labels
    :param dataset: the data to split
    :param lookback: the number of days to consider as training data
    :return: X and y
    """
    x_data, y_data = [], []
    for i in range(len(dataset) - lookback - 1):
        x_data.append(dataset[i:i + lookback, 0])
        y_data.append(dataset[i + lookback, 0])
    return np.array(x_data), np.array(y_data)


def train_test_split(
        dataset: pd.DataFrame,
        lookback: int,
        train_size: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing
    :param dataset: the data to split
    :param lookback: the number of days to consider as training data
    :param train_size: size of training data (in percentage)
    :return: x_train, x_test, y_train, y_test
    """
    # Computing split index based on train size
    split_index = int(len(dataset) * train_size)
    # Splitting train data and test data
    train_data, test_data = dataset[0:split_index, :], dataset[split_index:len(dataset), :]
    # Splitting each dataset into train and test
    x_train, y_train = split_data(dataset=train_data, lookback=lookback)
    x_test, y_test = split_data(dataset=test_data, lookback=lookback)
    # Returning datasets
    return x_train, x_test, y_train, y_test


def model_evaluation(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> float:
    """
    Evaluates the model based on stock movement
    :param y_true: True stock values
    :param y_pred: Predicted stock values
    :return: Metric
    """
    # Initialising counter to 0
    count = 0
    for i in range(1, len(y_pred)):
        # If true value was bullish and predicted value is bullish
        if y_true[i] > y_true[i - 1] and y_pred[i] > y_true[i - 1]:
            # Increment counter
            count += 1
        # If true value was bearish and predicted value is bearish
        elif y_true[i] < y_true[i - 1] and y_pred[i] < y_true[i - 1]:
            # Increment counter
            count += 1
    # Return percentage of correct movements in all predictions
    return count / len(y_pred)


def model_selection(
        ticker: str,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metric: Literal['r2_score', 'direction', 'mean_squared_error'],
        show_plots: bool = False
) -> tuple[Sequential, np.ndarray, float] | tuple[None, None, None]:
    """
    Test multiple sequential sequential_models and returns the best performing one
    :param ticker: Stock ticker symbol
    :param x_train: training data
    :param y_train: training labels
    :param x_test: testing data
    :param y_test: testing labels
    :param metric: Metric to test
    :param show_plots: Whether to show the plots
    :return: best model
    """
    # Initialising search
    if metric in ['r2_score', 'direction']:
        best_movement_estimation = 0
    elif metric == 'mean_squared_error':
        best_movement_estimation = 1
    best_model = None
    best_model_predictions = None
    # Iterating over number of sequential_models
    print(f'Starting model selection for {ticker}:')
    for n_models in range(max_layers):
        # Instantiating model
        model = Sequential()
        model._name = f'model_{n_models}'
        # Adding corresponding layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        for n_layers in range(n_models):
            model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        # Compiling model
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(f'model_{n_models} -> created')
        # Fitting model to data
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, verbose=False)
        print(f'model_{n_models} -> fitted')
        # Predicting test data and evaluating model
        y_pred = model.predict(x=x_test, verbose=False)
        if metric == 'r2_score':
            current_model_performance = r2_score(y_true=y_test.flatten(), y_pred=y_pred.flatten())
        elif metric == 'direction':
            current_model_performance = model_evaluation(y_true=y_test.flatten(), y_pred=y_pred.flatten())
        elif metric == 'mean_squared_error':
            current_model_performance = mean_squared_error(y_true=y_test.flatten(), y_pred=y_pred.flatten())
        if show_plots:
            plot_prediction(
                ticker=ticker,
                model=model,
                y_true=y_test.flatten(),
                y_pred=y_pred.flatten(),
                show=True,
                score=current_model_performance
            )
        print(f'model_{n_models} -> Score: {current_model_performance:.5f}')
        print('-------------------------')
        if metric in ['r2_score', 'direction']:
            # Storing best performing model
            if best_movement_estimation < current_model_performance <= 1:
                best_movement_estimation = current_model_performance
                best_model = model
                best_model._name = model.name
                best_model_predictions = y_pred
        elif metric == 'mean_squared_error':
            if 0 <= current_model_performance < best_movement_estimation:
                best_movement_estimation = current_model_performance
                best_model = model
                best_model._name = model.name
                best_model_predictions = y_pred
    # Returning best performing model
    if best_model is not None:
        print(f'Selected: {best_model.name} Score: {best_movement_estimation:2f}')
        # Plot the model
        plot_model(
            model=best_model,
            to_file=f'../graphs/sequential_models/{ticker}.png',
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True
        )
        return best_model, best_model_predictions, best_movement_estimation
    return None, None, None


def plot_prediction(
        ticker: str,
        model: Sequential,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        score: float,
        show: bool = False
) -> None:
    """
    Plots the predictions of the stock values
    :param ticker: Stock ticker symbol
    :param model: Model used for predictions
    :param y_true: true values of the stock closing price
    :param y_pred: predicted values of the stock closing price
    :param score: score of the model
    :param show: whether to show the plot or not
    """
    plt.figure(figsize=(20, 10))
    sns.lineplot(x=list(range(len(y_true))), y=y_true, label='true data')
    sns.lineplot(x=list(range(len(y_pred))), y=y_pred, label='predictions')
    plt.title(label=f'Prediction for {ticker} by {model.name}. {prediction_metric}: {score}', fontweight='bold')
    if not show:
        # Check if the path exists
        if not os.path.exists('../graphs/prediction/'):
            # Create the directory
            os.makedirs('../graphs/prediction/')
        plt.tight_layout()
        plt.savefig(f'../graphs/prediction/{ticker}.png')
        plt.close()
    else:
        plt.show()


def filter_portfolio_evolution(pf_values: dict) -> dict:
    """
    Remove duplicate values and dates from portfolio evolution data
    :param pf_values:
    :return:
    """
    df = pd.DataFrame(pf_values)
    df = df.drop_duplicates(subset='date', keep='last')
    return df.to_dict(orient='list')
