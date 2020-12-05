import easygui
import tkinter as tk
from scipy.stats import norm
from datetime import datetime, date, time, timedelta
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import statistics
import sklearn
import math
from sklearn import metrics
from matplotlib import pyplot
from pandas import read_csv
from sklearn.utils import shuffle
import os
import ctypes
from keras.layers import Input, Dense, Dropout, concatenate, Conv1D, MaxPooling1D, Flatten, AveragePooling1D
from keras.models import Model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from util.enums import GranularityLevel,InputEnums
#from input_lag_lead import input_lead_lag



class App(tk.Frame):
    def __init__(self,master=None,**kw):
        #Create a blank dictionary
        self.var = {}
        self.answers = {}
        tk.Frame.__init__(self,master=master,**kw)

        tk.Label(self,text="Indicator Benchmark Value").grid(row=0,column=0)
        self.question1 = tk.Entry(self)
        self.question1.grid(row=0,column=1)

        tk.Label(self,text="Indicator Good Value ( <= Deviation in +/- %)").grid(row=1,column=0)
        self.question2 = tk.Entry(self)
        self.question2.grid(row=1,column=1)

        tk.Label(self, text="Low Limit 1 ").grid(row=2, column=0)
        self.question3 = tk.Entry(self)
        self.question3.grid(row=2, column=1)

        tk.Label(self, text="Low Limit 2 ").grid(row=3, column=0)
        self.question4 = tk.Entry(self)
        self.question4.grid(row=3, column=1)

        tk.Label(self, text="Granularity(w,d,h,m,s,tm:TenMinute,ts:ThirtySec): ").grid(row=4, column=0)
        self.question5 = tk.Entry(self)
        self.question5.grid(row=4, column=1)

        tk.Label(self, text="Start Date (format: yyyy-mm-dd: 2020-12-31)").grid(row=5, column=0)
        self.question6 = tk.Entry(self)
        self.question6.grid(row=5, column=1)

        tk.Label(self, text="End Date (format: yyyy-mm-dd: 2020-12-31)").grid(row=6, column=0)
        self.question7 = tk.Entry(self)
        self.question7.grid(row=6, column=1)

        tk.Label(self, text="Enter Filename to save features as .CSV ").grid(row=7, column=0)
        self.question8 = tk.Entry(self)
        self.question8.grid(row=7, column=1)

        try:
            tk.Button(self,text="Go",command = self.collectAnswers).grid(row=8,column=1)
        except ValueError as e:
            ctypes.windll.user32.MessageBoxW(0, str(e)+"\n\n Rerun Code to enter values correctly", "ERROR MESSAGE", 1)
            exit(0)


    def collectAnswers(self):
        self.answers['IndicatorBenchmarkValue'] = self.question1.get()
        self.answers['IndicatorGoodValue'] = self.question2.get()
        self.answers['LowLimit1Value'] = self.question3.get()
        self.answers['LowLimit2Value'] = self.question4.get()
        self.answers['Granularity'] = self.question5.get()
        self.answers['SDate'] = self.question6.get()
        self.answers['EDate'] = self.question7.get()
        self.answers['FileName'] = self.question8.get()
        printAnswers(self.answers)
        ROOT_DIR = os.path.abspath(os.curdir)
        ctypes.windll.user32.MessageBoxW(0,"Generating Synthetic TimeSeries Data......\n")
        time.sleep(5)
        ctypes.windll.user32.MessageBoxW(0, "Synthetic Random Data Generation Done!\n\n File Created at location:\n"+ str(ROOT_DIR), "Synthetic Data Generation", 1)
        exit(0)

#
def printAnswers(answers):
    print("Indicator Benchmark Value: ", answers['IndicatorBenchmarkValue'])
    print("Indicator Good Value: ", answers['IndicatorGoodValue'])
    print("Low Limit1 Value: ", answers['LowLimit1Value'])
    print("Low Limit2 Value: ", answers['LowLimit2Value'])
    print("Granularity: ", answers['Granularity'])
    print("Start Date: ", answers['SDate'])
    print("End Date: ", answers['EDate'])
    print("FileName: ", answers['FileName']+'.csv')
    MaxVal_bm = float(answers['IndicatorBenchmarkValue']) + (
                 (float(answers['IndicatorGoodValue']) / 100) * float(answers['IndicatorBenchmarkValue']))
    MinVal_bm = float(answers['IndicatorBenchmarkValue']) - (
                 (float(answers['IndicatorGoodValue']) / 100) * float(answers['IndicatorBenchmarkValue']))

    print(MinVal_bm)
    print(MaxVal_bm)
    SyntheticValuesCount = 0

    try:
        sdate = datetime.strptime(answers['SDate'], "%Y-%m-%d")
        edate = datetime.strptime(answers['EDate'], "%Y-%m-%d")
        deltadate = sdate - edate
        deltadate = abs(deltadate)
        print(deltadate.days)
    except ValueError as e:
        ctypes.windll.user32.MessageBoxW(0,str(e),"ERROR MESSAGE", 1)
        print(e)
        exit(0)
    #"Date Format entered wrong: Rerun Code with following format\n\n 'yy-mm-dddd'"


    if answers['Granularity'] == 'd':
        SyntheticValuesCount = deltadate.days
        level = GranularityLevel.one_day.value

    elif answers['Granularity'] == 'h':
        SyntheticValuesCount = (deltadate.days) * 24
        level = GranularityLevel.one_hour.value

    elif answers['Granularity'] == 'm':
        SyntheticValuesCount = (deltadate.days) * 24 * 60
        level = GranularityLevel.one_minute.value

    elif answers['Granularity'] == 's':
        SyntheticValuesCount = (deltadate.days) * 24 * 60 * 60
        level = GranularityLevel.one_sec.value

    elif answers['Granularity'] == 'tm':
        SyntheticValuesCount = (deltadate.days) * 24 * 6
        level = GranularityLevel.ten_min.value

    elif answers['Granularity'] == 'ts':
        SyntheticValuesCount = (deltadate.days) * ((24 * 60) * 2)
        level = GranularityLevel.thirty_sec.value

    elif answers['Granularity'] == 'w':
        SyntheticValuesCount = (deltadate.days) / 7
        level = GranularityLevel.one_week.value

    else:
        print("Granularity entered wrong: Rerun Code with following entry:\n\n"+
                  "h:hourly\n"+
                  "s: Second\n"+
                  "d: daily\n"+
                  "w:weekly\n"+
                  "ts:thirty seconds\n"+
                  "tm:ten minutes\n")
        ctypes.windll.user32.MessageBoxW(0, "Granularity entered wrong: h:hourly\n s:second\n d:daily\n w:weekly\n ts:thirty seconds\n tm:ten minutes\n", "ERROR MESSAGE", 1)
        exit(0)


    benchmarkrange = math.ceil(float(0.50 * SyntheticValuesCount))  # 60% values from within benchmark defined
    limitrange     = math.ceil(int(0.30 * SyntheticValuesCount))   # 35% values from within low limit 1 & 2 defined
    outliersrange1  = int(0.05 * SyntheticValuesCount)   # 5% values as outliers or alarm values.
    outliersrange2 =  int(0.05 * SyntheticValuesCount)  # 5% values as outliers or alarm values.
    replicate_range = int(0.10 * SyntheticValuesCount)

    Generated_bm_Values = np.random.uniform(MinVal_bm, MaxVal_bm, benchmarkrange)
    #Generated_bm_Values_dup = [np.random.randrange(MinVal_bm, MaxVal_bm) for i in range(benchmarkrange)]
        #np.random.choice(MinVal_bm, MaxVal_bm, benchmarkrange, replace=True)


    Generated_lm_Values = np.random.uniform(float(answers['LowLimit2Value']), float(answers['LowLimit1Value']), limitrange)
    Generated_out1_Values = np.random.uniform(float(MaxVal_bm+1),float(MaxVal_bm+1.5) , outliersrange1)
    Generated_out2_Values = np.random.uniform(float(float(answers['LowLimit2Value'])-0.5), float(float(answers['LowLimit2Value']) - 1.5), outliersrange2)


    #np.concatenate((a, b), axis=0)

    SyntheticData = np.concatenate((Generated_bm_Values, Generated_lm_Values, Generated_out1_Values, Generated_out2_Values), axis=None)
    df = pd.DataFrame(data=SyntheticData, columns=["Feature"])

    if SyntheticValuesCount == benchmarkrange + limitrange + outliersrange1 + outliersrange2:
        SyntheticData = np.concatenate((Generated_bm_Values, Generated_lm_Values, Generated_out1_Values, Generated_out2_Values), axis=None)

    elif SyntheticValuesCount < (benchmarkrange + limitrange + outliersrange1 + outliersrange2):
        df.drop(df.tail(SyntheticValuesCount - (benchmarkrange + limitrange + outliersrange1 + outliersrange2)).index,inplace=True)

    else:
        for i in range(abs((benchmarkrange + limitrange + outliersrange1 + outliersrange2 + replicate_range) - SyntheticValuesCount)):
            #df = df.append({'Feature': np.random.uniform(MinVal_bm, MaxVal_bm, 1) }, ignore_index=True)
            df = df.append(pd.DataFrame(np.random.uniform(MinVal_bm, MaxVal_bm, 1), columns=['Feature']),ignore_index=True)

    df_replicate = df.head(replicate_range)
    df = df.append(df_replicate, ignore_index=True)
 #   print("original dataframe:\n")
    print(df)
    df = shuffle(df)
    print("shuffled data frame:\n")
    print(df)
    #df_repeat = df.sample(n=len(df), replace=True)

    #print(df_repeat)
    #print(SyntheticData)
    # print([item for item, count in collections.Counter(df).items() if count > 1])
    # print(df.values)
    # df.to_csv(answers['FileName']+'.csv' , index= False)
    # print([item for item, count in collections.Counter(df).items() if count > 1])

    df['ts'] = pd.DataFrame({'ts': pd.date_range(start=sdate, end=edate, freq=get_freq_by_level(level))})


    df['ts'] = pd.to_datetime(df['ts'])
    df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
    df = df.sort_values(by='ts')
    df.to_csv(answers['FileName'] + '.csv', index=False)
    print(df)

    series = read_csv(answers['FileName'] + '.csv', header=0, index_col=1)
    print(series.shape)

    #print(series)
    #X[0, :], X[1, :], c = y, cmap = plt.cm.Spectral
    # pyplot.plot(series)
    # pyplot.show()
    #
    # input_lead_lag()
    # lead_time, lag_time = send_values(lead=0,lag=0)

    # choice = easygui.enterbox("Enter Machine Learning Model choice:\n 1:MultiLayerPerceptron\n 2:Convolutional Neural Network\n 3:CNN_with_HyperParameterTuning\n 4:MLP_with_HyperParameterTuning\n Default:MultiLayerPerceptron")
    # # choice = input("Enter choice:\n 1:mlp\n 2:cnn\n 3:cnn_hp\n 4:mlp_hp")
    # print("User Choice is: " + choice)



    features = get_features(InputEnums.input.value)
    data = get_data(series,sdate,edate,level)
    forecast_input = get_forecast_input(data['Feature'],features, InputEnums.lag_time_steps.value)
    #data = series_to_supervised(series, 3)

    x_inputs, x_outputs, testX, testY, trainY = prepare_data_mlp(data['Feature'], features, InputEnums.lag_time_steps.value, InputEnums.lead_time_steps.value, InputEnums.test_train_split_size.value)

    model = get_mlp_model(x_inputs, InputEnums.lag_time_steps.value, InputEnums.lead_time_steps.value)

    forecast, model, rmse_list, rmse = get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX,
                                                                       testY,
                                                                       InputEnums.lead_time_steps.value,
                                                                       InputEnums.lag_time_steps.value, features, forecast_input,
                                                                       InputEnums.confidence_interval_multiple_factor.value)

    print("end")

    #series = read_csv('monthly-car-sales.csv', header=0, index_col=0)




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

def get_data(df, start_period, end_period, granularity_level = GranularityLevel.one_hour.value[0],immute= True, drop_nan= True):
    df.sort_values(by=['ts'], inplace=True)
    res_df = df.drop_duplicates().reset_index(drop=True)

    if immute:
        res_df = res_df.fillna(method='ffill')
    if drop_nan:
        res_df.dropna(inplace=True)

        return res_df


def rmse_time_series(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(metrics.mean_squared_error(y_true[:,i], y_pred[:,i]))
        rmses.append(rmse)
    return sum(rmses)/len(rmses), rmses


def send_values(lead,lag):
    return lead,lag
#
# def train_test_split(data, n_test):
# 	return data[:-n_test], data[-n_test:]

#
#
# def 1_series_to_supervised(data, n_in=1, n_out=1):
# 	df = DataFrame(data)
# 	cols = list()
# 	# input sequence (t-n, ... t-1)
# 	for i in range(n_in, 0, -1):
# 		cols.append(df.shift(i))
# 	# forecast sequence (t, t+1, ... t+n)
# 	for i in range(0, n_out):
# 		cols.append(df.shift(-i))
# 	# put it all together
# 	agg = concat(cols, axis=1)
# 	# drop rows with NaN values
# 	agg.dropna(inplace=True)
# 	return agg.values


def get_freq_by_level(ganuality_level_value):
    if GranularityLevel.one_hour.value[0] == ganuality_level_value[0]:
        return '60T'
    elif GranularityLevel.three_hour.value[0] == ganuality_level_value[0]:
        return '180T'
    elif GranularityLevel.one_day.value[0] == ganuality_level_value[0]:
        return '1440T'
    elif GranularityLevel.one_week.value[0] == ganuality_level_value[0]:
        return '10080T'
    elif GranularityLevel.ten_min.value[0] == ganuality_level_value[0]:
        return '10T'
    elif GranularityLevel.one_minute.value[0] == ganuality_level_value[0]:
        return '1T'
    elif GranularityLevel.one_sec.value[0] == ganuality_level_value[0]:
        return '0.016666666667T'
    elif GranularityLevel.thirty_sec.value[0] == ganuality_level_value[0]:
        return '0.5T'





def generatedata(answers):
    CountOfRandValues = 100
    OutliersCount = int(CountOfRandValues / 10)
    MaxFeature = answers['IndicatorBenchmarkValue'] + (
                (answers['IndicatorGoodValue'] / 100) * answers['IndicatorBenchmarkValue'])
    MinFeature = answers['IndicatorBenchmarkValue'] - (
                (answers['IndicatorGoodValue'] / 100) * answers['IndicatorBenchmarkValue'])
    print(MinFeature)
    print(MaxFeature)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Synthetic Data Generation")
    windowWidth = root.winfo_reqwidth()
    windowHeight = root.winfo_reqheight()
    print("Width", windowWidth, "Height", windowHeight)

    # Gets both half the screen width/height and window width/height
    positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2)
    positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2)

    # Positions the window in the center of the page.
    root.geometry("+{}+{}".format(positionRight, positionDown))
    #root.geometry("".format(positionRight, positionDown))
    App(root).grid()
    root.mainloop()
