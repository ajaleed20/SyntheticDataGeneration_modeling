import pandas as pd
import numpy as np
from sklearn import metrics
from keras.layers import Input, Dense, Dropout, concatenate, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.models import Model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split



def get_mlp_model(x_inputs, lag_time_steps, lead_time_steps):
    input_models = []
    dense_layers = []
    for i in range(len(x_inputs)):
        visible = Input(shape=(lag_time_steps,))
        hidden0 = Dense(100, activation='relu')(visible)  # sigmoid #tanh #relu
        dropout0 = Dropout(0.5)(hidden0)
        hidden1 = Dense(50, activation='relu')(dropout0)  # sigmoid #tanh #relu
        dropout1 = Dropout(0.5)(hidden1)
        dense = Dense(50, activation='relu')(dropout1)  # relu #sigmoid #tanh
        input_models.append(visible)
        dense_layers.append(dense)

    if len(x_inputs) > 1:
        merge = concatenate(dense_layers)
    else:
        merge = dense_layers[0]

    hidden1 = Dense(len(x_inputs) * 16, activation='relu')(merge)  # sigmoid #tanh #relu
    dropout1 = Dropout(0.1)(hidden1)
    hidden2 = Dense(len(x_inputs) * 8, activation='relu')(dropout1)  # relu #sigmoid #tanh
    dropout2 = Dropout(0.5)(hidden2)
    hidden3 = Dense(len(x_inputs), activation='relu')(dropout2)  # relu #sigmoid #tanh
    dropout3 = Dropout(0.1)(hidden3)
    output = Dense(lead_time_steps)(dropout3)

    model = Model(inputs=input_models, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_data_mlp(data, features, lag_time_steps, lead_time_steps, test_split_size):
    X, y = series_to_supervised(data, lag_time_steps, lead_time_steps)

    leadColumns = [col for col in y.columns] #if target_mp_id == col[0:col.find('(')]]
    y = y[leadColumns]
    values = X.values
    X = values.reshape((values.shape[0], lag_time_steps, len(features)))
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split_size, shuffle=False)

    x_inputs = []
    x_outputs = []

    for i in range(trainX.shape[2]):
        x_inputs.append(trainX[:, :, i-1])
    for i in range(testX.shape[2]):
        x_outputs.append(testX[:, :, i-1])

    return x_inputs, x_outputs, testX, testY, trainY




def get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX, testY, lead_time_step, lag_time_steps,
                                    features, forecast_input, confidence_interval_multiple_factor ,normalize_data= False,  normalizer= None, power_transformers= None):

        history = model.fit(x_inputs, trainY, epochs=300, verbose=2, validation_data=(x_outputs, testY))
        # plt.plot(history.history['loss'], label='train')
        # plt.plot(history.history['val_loss'], label='test')
        # plt.legend()
        # plt.savefig('Graphs/'+'specialties_ '+'_learning curve_Lag_' + str(lag_time_steps) + '_Lead_' + str(lead_time_step) + '_iteration')
        # plt.show()
        forecast = model.predict(forecast_input)

        rmse, rmse_list = cal_rmse(model, x_outputs, testX, testY, lead_time_step,lag_time_steps, features, normalizer,
                                   power_transformers, normalize_data=False)
        rmse_list = [rmse * confidence_interval_multiple_factor for rmse in rmse_list]

        return forecast, model, rmse_list,rmse


def get_features(var):
    return list(map(str, var))


def get_forecast_input(data,features , lag_time_steps):
    X, y = series_to_supervised(data, lag_time_steps, 0)
    values = X.values
    X = values.reshape((values.shape[0], lag_time_steps, len(features)))
    x_inputs = []

    for i in range(X.shape[2]):
        x_inputs.append(np.reshape(X[-1, :, i-1], (1, X.shape[1])))
    return x_inputs


def cal_rmse(model, x_outputs, testX, testY, lead_time_steps,
             lag_time_steps, features, normalizer, power_transformers, normalize_data=True):
    forecast = model.predict(x_outputs)
    testY_transformed = testY.values

    if normalize_data:
        tempTestX = testX.reshape(testX.shape[0], lag_time_steps * len(features))[:,
                        -(len(features)):-1]
        if len(tempTestX.shape) == 1:
            tempTestX = np.reshape(tempTestX, (1, tempTestX.shape[0]))

        for i in range(lead_time_steps):
            raw_forecast = inverse_transform_forecast(forecast[:, i], tempTestX, normalizer, power_transformers,
                                                          features)
            forecast[:, i] = raw_forecast.reshape(1, -1)

        for i in range(lead_time_steps):
            raw_test_data = inverse_transform_forecast(testY_transformed[:, i], tempTestX, normalizer, power_transformers,
                                                          features)
            testY_transformed[:, i] = raw_test_data.reshape(1, -1)

    rmse, rmse_list = rmse_time_series(testY_transformed, forecast)

    return rmse, rmse_list


def rmse_time_series(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(metrics.mean_squared_error(y_true[:,i], y_pred[:,i]))
        rmses.append(rmse)
    return sum(rmses)/len(rmses), rmses


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    data = pd.DataFrame(data)
    n_vars = 1 if type(data) is list else data.shape[1]
    columns = data.columns
    df = pd.DataFrame(data)
    cols, leadNames, lagNames = list(), list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        lagNames += [(columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        leadNames += [(columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]

    res = pd.concat(cols, axis=1)
    res.columns = np.concatenate((lagNames, leadNames))

    # drop rows with NaN values
    if dropnan:
        res.dropna(inplace=True)

    return res[lagNames], res[leadNames]



def inverse_transform_forecast(pred_y, test_x, scaler, power_transformers, features):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    raw_pred_y = pred_y.reshape(-1, 1)
    test_x = np.concatenate((test_x, raw_pred_y), axis=1)
    for i in range(len(features)):
        if features[i] in power_transformers:
            test_x[:, i] = power_transformers[features[i]].inverse_transform(test_x[:, i].reshape(-1, 1)).reshape(test_x.shape[0])
    raw_pred_y = scaler.inverse_transform(test_x)[:, -1].reshape(-1, 1)

    if np.isnan(raw_pred_y).any():
        imp.fit(raw_pred_y)
        raw_pred_y = imp.transform(raw_pred_y)

    return raw_pred_y

def pre_process_data(df,immute= True, drop_nan= True):

    if immute:
        df = df.fillna(method='ffill')
    if drop_nan:
        df.dropna(inplace=True)

        return df


def rmse_time_series(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(metrics.mean_squared_error(y_true[:,i], y_pred[:,i]))
        rmses.append(rmse)
    return sum(rmses)/len(rmses), rmses

def executeRegression(features, data, lag_time_steps, lead_time_steps, train_test_split, confidence_interval_multiple_factor):

    data = pre_process_data(data)
    forecast_input = get_forecast_input(data['Feature'], features, lag_time_steps)

    x_inputs, x_outputs, testX, testY, trainY = prepare_data_mlp(data['Feature'], features,
                                                                 lag_time_steps,
                                                                 lead_time_steps,
                                                                 train_test_split)

    model = get_mlp_model(x_inputs, lag_time_steps, lead_time_steps)

    forecast, model, rmse_list, rmse = get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX,
                                                                       testY,
                                                                       lead_time_steps, lag_time_steps, features, forecast_input, confidence_interval_multiple_factor)

    return forecast, model, rmse_list, rmse