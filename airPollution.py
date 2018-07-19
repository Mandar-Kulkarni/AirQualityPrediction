# Package Imports

import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics, cross_validation
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools
import numpy as np
from  statsmodels.tsa.arima_model import ARIMA


# Reading csv in to pandas dataframe

def read_data():

    data = pd.read_csv('AirPollution.csv')

    return data


# Data Cleaning for Classification Methods


def gaussianNB(features, labels):

    acc = []
    classifier = GaussianNB()
    predict = cross_validation.cross_val_predict(classifier, features, labels, cv=10)
    acc.append(metrics.accuracy_score(predict, labels))
    F1 = metrics.f1_score(labels, predict, average='micro')
    P = metrics.precision_score(labels, predict, average='micro')
    R = metrics.recall_score(labels, predict, average='micro')
    return (float(sum(acc) / len(acc)))*100, F1*100, P*100, R*100

def DT(feature, labels):

    acc = []
    classifier = DecisionTreeClassifier()
    predict = cross_validation.cross_val_predict(classifier, features, labels, cv=10)
    acc.append(metrics.accuracy_score(predict, labels))
    F1 = metrics.f1_score(labels, predict, average='micro')
    P = metrics.precision_score(labels, predict, average='micro')
    R = metrics.recall_score(labels, predict, average='micro')
    return (float(sum(acc) / len(acc)))*100, F1*100, P*100, R*100

def pm_classification(data):

    good = []
    moderate = []
    unhealthysg = []
    unhealthy = []
    pm_labels = []
    bad_values = []
    pm_list = data['pm2.5']

    for i, value in enumerate(pm_list):

        if math.isnan(value):
            #del pm_list[i]
            bad_values.append(i)
        elif (value > 0 and value <= 50):
            good.append(i)
            pm_labels.append(0)
        elif (value > 50 and value <= 100):
            moderate.append(i)
            pm_labels.append(1)
        elif (value > 100 and value <= 150):
            unhealthysg.append(i)
            pm_labels.append(2)
        else:
            unhealthy.append(i)
            pm_labels.append(3)


    data.drop(data.index[bad_values], inplace=True)
    features = data[['Iws', 'TEMP', 'PRES', 'Is', 'Ir']].copy()

    return features, pm_labels


def ar1(phi=0.9, n=1000, init=0):
    time_series = [init]
    error = np.random.rand(n)
    for period in range(n):
        time_series.append(error[period]+phi*time_series[-1])
    return pd.Series(time_series[1:], index=range(n))


def ar2(phi1=0.9, phi2=-0.8, n=1000, init=0):
    time_series = [init, init]
    error = np.random.rand(n)
    for period in range(2, n):
        time_series.append(error[period]+phi1*time_series[-1]+phi2*time_series[-2])
    return pd.Series(time_series[1:], index=range(1,n))

def MA1(theta=0.5, n=100):
    time_series = []
    error=np.random.rand(n)
    for period in range(1,n):
        time_series.append(error[period]+ theta*error[period-1])
    return pd.Series(time_series[1:], index=range(1, n-1))

if __name__ == "__main__":

    # Read Data and convert the columns to DateTime index

    data = read_data()
    date_time = pd.DataFrame(pd.to_datetime(data[['year', 'month', 'day', 'hour']]))
    data = pd.concat([date_time, data], axis=1)
    data = data.drop(['year', 'No', 'month', 'day', 'hour'], axis=1)
    data.columns.values[0]='Time'
    data.set_index(['Time'], inplace=True)

    # Drop Wind Direction values
    data = data.drop(['cbwd'], axis=1)

    # Drop NaN Values from PM2.5 Data
    data['pm2.5'].dropna(inplace=True)

    # Generate Auto-Correlation and Partial Auto-Correlation Graphs

    acf_results = statsmodels.tsa.stattools.acf(data['pm2.5'])
    pacf_results = statsmodels.tsa.stattools.pacf(data['pm2.5'])

    plt.plot(acf_results, color='green', label='Auto-Correlation', linestyle='--')
    plt.plot(pacf_results, color='red', label='Partial Auto-Correlation')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

    # Consider data for the year 2010 only

    data = data['2010']

    # Data differencing example for introducing Stationarity

    log_temp = np.log(data['pm2.5'])
    log_temp_diff = log_temp - log_temp.shift()
    log_temp_diff.plot()

    # Remove NaN values

    log_temp_diff.dropna(inplace=True)
    features, pm_labels = pm_classification(data)

    acc_NB, F_NB, precision_NB, recall_NB = gaussianNB(features, pm_labels)
    acc_DT, F_DT, precision_DT, recall_DT = DT(features, pm_labels)
    # ARIMA model

    # order = (p,d,q) p -AR, q -MA d -diff
    data['pm2.5'].dropna(inplace=True)
    model = ARIMA(data['pm2.5'], order=(2,1,0))
    results_AR = model.fit(disp=-1)
    plt.plot(log_temp_diff)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.show()


    # SVM(features, pm_labels)

    '''
    dftest = adfuller(log_temp_diff, autolag='AIC', regression='c')[:5]
    useful_values = [v for v in dftest[:4]]
    useful_values.extend([dftest[4]['1%'], dftest[4]['5%'], dftest[4]['10%']])
    adFuller = pd.DataFrame({'Value': useful_values, 'Label':['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']})
    
    print(adFuller)
    '''






# ARIMA

    # Munging and Stationary

    # Difference of the data from mean for stationarity
    # OR Linear Regression
    # Remove Seasonality

    # TO check Stationarity Dickey Fuller Test

    # ARIMA - Moving Average is a stationary process with only one prior time and error of previous and current time step
    # AutoRegressive Process -  value at current time depends on the previous value
    # I is for differencing - done by algorithm. optional.


    # Use ACF(MA) and PACF (AR) for finding thr order of ARIMA

    # PACF used for AR because the current value only depends on one previous step and not on previous step value

    # Higher order is overfitting




# Prediction



    # Temperature trends over the entire year. How periodic the data is!

    '''log_temp = np.log(data['TEMP'])
    log_temp_diff = log_temp - log_temp.shift()
    log_temp_diff.plot()
    plt.show()
    '''
    # self correlation and auto correlation

    # partial auto correlation avoids harmonics in the lag and uses shorter correlation

# Divide second value in the window with the 3rd value(Used for custom weighting)

#    r = data.rolling(window=100, center=False).apply(lambda x: x[1]/x[2])[100:200]
#    print(r)



    #pm_classification(data)

    #print(pd.to_datetime(data[['year', 'month', 'day', 'hour']]))


    # Window functions
    '''
    r = data.rolling(window=100)
    data['pm2.5'].plot(color='gray')
    r.mean()['pm2.5'].plot(color='red')
    plt.show()'''



    # data_diff = data.diff()
    # print(data_diff)

    # data_lagged = data.shift()

    # print(data_lagged)

    #test_stationarity(data['pm2.5'])
    # plt.plot(data['pm2.5'])
    # plt.show()

# Lag Function - Move the function back and forward in time. Time shifting
# Window Function -  Rolling window(Sampling and has a lag) and Expanding Window(Cumulative Summation from the beginning)


# Plot of unhealthy concentration of PM2.5 per year

'''
    years = [2010, 2011, 2012, 2013, 2014]
    hour_plot = []
    pm_plot = []
    data.set_index(['year'], inplace=True)

    for j, k in enumerate(years):
        pm_plot.append(0);
        for i in range(len(data)):
            if (data.iloc[i]['pm2.5'] > 150 and k == data.index[i]):
                print(data.index[i])
                pm_plot[j] += 1

    plt.bar(years, pm_plot, )
    plt.title('High PM conc. hrs. per year')
    plt.xlabel('Year')
    plt.ylabel('Total hours for PM > 150')
    plt.show()

'''





