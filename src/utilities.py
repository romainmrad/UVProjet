import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score


def split_data(dataset, time_step):
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        x_data.append(dataset[i:(i + time_step), 0])
        y_data.append(dataset[i + time_step, 0])
    return np.array(x_data), np.array(y_data)


def train_test_split(dataset, time_step, train_size=0.7):
    # Computing split index based on train size
    split_index = int(len(dataset) * train_size)
    # Splitting train data and test data
    train_data, test_data = dataset[0:split_index, :], dataset[split_index:len(dataset), :]
    # Splitting each dataset into train and test
    x_train, y_train = split_data(dataset=train_data, time_step=time_step)
    x_test, y_test = split_data(dataset=test_data, time_step=time_step)
    # Returning datasets
    return x_train, x_test, y_train, y_test


def model_selection(
        models_dict: dict[str, dict[str, Sequential | float]]
) -> Sequential:
    """
    Finds the best model and returns it
    :param models_dict: dictionary of models
    :return: best model
    """
    best_model = None
    closest_to_1_diff = float('inf')  # Initialize with positive infinity

    for model_name, model_info in models_dict.items():
        score = model_info["r2_score"]
        diff_to_1 = abs(1 - score)

        if diff_to_1 < closest_to_1_diff:
            closest_to_1_diff = diff_to_1
            best_model = model_info['model']

    return best_model


def model_evaluation(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
) -> Sequential:
    """
    Select the best Sequential model
    :param x_train: training data
    :param y_train: training labels
    :param x_test: testing data
    :param y_test: testing labels
    :return: best model
    """
    models = dict()
    for n_models in range(4):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        for n_layers in range(n_models):
            model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=64, verbose=True)
        predictions = model.predict(x_test)
        models[f'Model-{n_models}'] = dict()
        models[f'Model-{n_models}']['model'] = model
        models[f'Model-{n_models}']['r2_score'] = r2_score(y_test, predictions)
    return model_selection(models)

