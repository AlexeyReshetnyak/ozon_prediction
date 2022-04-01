#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from IPython.display import display # TODO: is it needed?
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def main():
    #TODO: fit all to 79 cols
    file = './data/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv'
    data = pd.read_csv(file)

    print("\nDrop some non useful columns.")
    #input("Press Enter to continue...")

    cols_to_drop = ['No', 'year', 'month', 'day', 'hour', 'wd', 'station']
    data = data.drop(cols_to_drop, axis=1)
    data.info()

    print('Are there any duplicated values in data ? : {}\n'.format(data.duplicated().any()))
    print('The total number of null values in each colum:')
    display(data.isnull().sum())
    data.fillna(value=data.mean(), inplace=True)
    display(data.isnull().any())

# TODO: dicede what plot is needed
#    plt.figure(figsize=(12,5))
#    sns.distplot(data['O3'], bins=50)
#    plt.title('Ozon dencity', fontsize=16)
#    plt.show()

    print("Lets see corellation between the features of the data")
    #input("Press Enter to continue...")
    plt.figure(figsize=(13,9))
    correlation_data = data[['PM2.5', 'PM10', 'SO2', 'NO2',
                             'CO', 'O3', 'TEMP', 'PRES',
                             'DEWP', 'RAIN', 'WSPM']]
    sns.heatmap(correlation_data.corr(),cmap=plt.cm.Reds,annot=True)
    plt.title('Heatmap displaying the correlation matrix of the variables',fontsize=16)
    plt.show()

    print("\nWe see that only two pairs of features correlate well.\n\
          PM 10, PM2.5 and TEMP, DEWP with coefficients 0.87, 0.82,\n\
          respectively. Not much to care about. Just ignore it.")
    #input("Press Enter to continue...")


    print("\nNow we will split data to predictor an outcome featires")
    #input("Press Enter to continue...")
    X = data.drop('O3', axis=1)
    y = data['O3'].to_numpy()
    X_scaled = preprocessing.scale(X)

    print("\nSplint to train and test. It's demanded by the task test data\
           \n is between 01.06.2016 to 30.11.2016, so split it manually\
           \n manually see nedeed period it's between 28513 and 32904 rows")
    #input("Press Enter to continue...")

    d1 = 28513; d2 = 32904
    X_test = X_scaled[d1:d2, :]
    y_test = y[d1:d2]

    X_train = np.concatenate((X_scaled[0:d1, :], X_scaled[d2:, :]), axis=0)
    y_train = np.concatenate((y[0:d1], y[d2:]), axis=0)


if __name__ == '__main__':
    main()
