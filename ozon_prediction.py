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

    plt.figure(figsize=(12,5))
    sns.distplot(data['O3'], bins=50)
    plt.title('Ozon dencity', fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()
