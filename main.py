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
from MLP_Regression import executeRegression
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

    # if SyntheticValuesCount == benchmarkrange + limitrange + outliersrange1 + outliersrange2:
    #     SyntheticData = np.concatenate((Generated_bm_Values, Generated_lm_Values, Generated_out1_Values, Generated_out2_Values), axis=None)
    #
    # el
    if SyntheticValuesCount < (benchmarkrange + limitrange + outliersrange1 + outliersrange2):
        df.drop(df.tail(SyntheticValuesCount - (benchmarkrange + limitrange + outliersrange1 + outliersrange2)).index,inplace=True)

    elif SyntheticValuesCount > (benchmarkrange + limitrange + outliersrange1 + outliersrange2):
        random_values = [];
        for i in range(abs((benchmarkrange + limitrange + outliersrange1 + outliersrange2 + replicate_range) - SyntheticValuesCount)):
            #df = df.append({'Feature': np.random.uniform(MinVal_bm, MaxVal_bm, 1) }, ignore_index=True)
            random_values.append(np.random.uniform(MinVal_bm, MaxVal_bm, 1));

        df = df.append(pd.DataFrame(random_values, columns=['Feature']),ignore_index=True)

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

    forecast, model, rmse_list, rmse = executeRegression(InputEnums.input.value,series,
                                                                       InputEnums.lag_time_steps.value, InputEnums.lead_time_steps.value, InputEnums.test_train_split_size.value,
                                                                       InputEnums.confidence_interval_multiple_factor.value)

    print("end")

    #series = read_csv('monthly-car-sales.csv', header=0, index_col=0)



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
