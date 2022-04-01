#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
from IPython.display import display # TODO: is it needed?
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    #TODO: fit all to 79 cols
    file = './data/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv'
    data = pd.read_csv(file)

    cols_to_drop = ['No','station']
    data = data.drop(cols_to_drop, axis=1)
    data.info()

    print('Are there any duplicated values in data ? : {}\n'.format(data.duplicated().any()))
    print('The total number of null values in each colum:')
    display(data.isnull().sum())
    data.fillna(value=data.mean(), inplace=True)
    data.wd.fillna(value='NE', inplace=True)
    display(data.isnull().any())

# TODO: dicede what plot is needed
#    plt.figure(figsize=(12,5))
#    sns.distplot(data['O3'], bins=50)
#    plt.title('Ozon dencity', fontsize=16)
#    plt.show()

    print("Lets see corellation between the features of the data")
    input("Press Enter to continue...")
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
    input("Press Enter to continue...")

if __name__ == '__main__':
    main()
