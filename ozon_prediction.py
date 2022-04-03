#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from IPython.display import display # TODO: is it needed?
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor

def main():

    """
    About the data. There are several files corresponding to different weather
    stations here. How to deal with them is not yet entirely clear. Maybe they
    need to be combined, maybe build models for each station separately.
    While we take some one station and work with it. For example file
    PRSA_Data_Aotizhongxin_20130301-20170228.csv.
    """

    file = './data/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv'
    #TODO: fit all to 79 cols
    #TODO: download, and unzip data from web
    data = pd.read_csv(file)

    # Drop some non useful columns.
    cols_to_drop = ['No', 'year', 'month', 'day', 'hour', 'wd', 'station']
    data = data.drop(cols_to_drop, axis=1)
    data.info()

    print('Are there any duplicated values in data ? : {}\n'.format(data.duplicated().any()))
    print('The total number of null values in each colum:')
    display(data.isnull().sum())
    data.fillna(value=data.mean(), inplace=True)
    display(data.isnull().any())

    # Let's do a little visualization
    plt.figure(figsize=(12, 5))
    sns.histplot(data['O3'], bins=100)
    plt.title('Ozon dencity', fontsize=16)
    plt.show()
    #It gives nothing special, data as data.

    # Lets see corellation between the features of the data
    plt.figure(figsize=(13, 9))
    correlation_data = data[['PM2.5', 'PM10', 'SO2', 'NO2',
                             'CO', 'O3', 'TEMP', 'PRES',
                             'DEWP', 'RAIN', 'WSPM']]
    sns.heatmap(correlation_data.corr(), cmap=plt.cm.Reds, annot=True)
    plt.title('Heatmap displaying the correlation matrix of the variables',
               fontsize=16)
    plt.show()

    """
    We see that only two pairs of features correlate well.
    PM 10, PM2.5 and TEMP, DEWP with coefficients 0.87, 0.82,
    respectively. Not much to care about. Just ignore it.
    """

    # Now we will split data to predictor, and outcome featires
    X = data.drop('O3', axis=1)
    y = data['O3'].to_numpy()

    # Preprocessin
    X_scaled = preprocessing.scale(X)

    """
    Split to train and test. It's demanded by the task test data
    is between 01.06.2016 to 30.11.2016, so split it
    manually see nedeed period. It's between 28513 and 32904 rows.
    """
    d1 = 28513; d2 = 32904
    X_test = X_scaled[d1:d2, :]
    y_test = y[d1:d2]

    X_train = np.concatenate((X_scaled[0:d1, :], X_scaled[d2:, :]), axis=0)
    y_train = np.concatenate((y[0:d1], y[d2:]), axis=0)
    
    # Let's try linear regression
    lin_model = LinearRegression()
    lin_model.fit(X_train,y_train)

    prediction = lin_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    accuracy = r2_score(y_test, prediction)

    print('Lenear regression Mean Squared Error (MSE): {}'.format(np.sqrt(mse)))
    print('Lenear regression model accuracy: {}\n'.format(accuracy))
    #Accuracy is about 0.6, very poor, let's try another model

    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, y_train)

    tree_pred = decision_tree.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_pred)
    tree_accuracy = r2_score(y_test, tree_pred)

    print('Decision tree Root Mean Squared Error: {}'.format(np.sqrt(tree_mse)))
    print('Decision tree model accuracy: {}\n'.format(tree_accuracy))
    #Accuracy is about the same as lin. regression, let's try another model

    forest = RandomForestRegressor(n_estimators=100,
                                   max_depth=7,
                                   max_features='auto',
                                   min_samples_split=7,
                                   min_samples_leaf=3)

    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    forest_mse = mean_squared_error(y_test, forest_pred)
    forest_accuracy = r2_score(y_test, forest_pred)

    print('Random forest Root Mean Squared Error: {}'.format(np.sqrt(forest_mse)))
    print('Random forest model accuracy: {}\n'.format(forest_accuracy))
    # Accuracy is about 0.74

    grad_boost = GradientBoostingRegressor(n_estimators=100,
                                           max_depth=7,
                                           max_features='auto',
                                           min_samples_split=7,
                                           min_samples_leaf=3,
                                           learning_rate=0.1)

    grad_boost.fit(X_train, y_train)

    gboost_pred = grad_boost.predict(X_test)
    gboost_mse = mean_squared_error(y_test, gboost_pred)
    gboost_accuracy = r2_score(y_test, gboost_pred)

    print('Gradient boosting Root Mean Squared Error: {}'.format(np.sqrt(gboost_mse)))
    print('Gradient boosting Overall model accuracy:{}\n'.format(gboost_accuracy))
    # Accuracy is about 0.76

    params = {'max_depth':[3,4,5,6,7,8,9],
              'max_features':['auto','sqrt','log2'],
              'min_samples_split':[2,3,4,5,6,7,8,9,10],
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}
    params['learning_rate'] = np.linspace(0.1, 1, 10)

    gradient_boosting = GradientBoostingRegressor()
    gboost_search = RandomizedSearchCV(gradient_boosting, params, n_jobs=-1,
                                       cv=5, verbose=1)
    gboost_search.fit(X_train, y_train)

    gboost_search_pred = gboost_search.predict(X_test)
    gboost_search_mse = mean_squared_error(y_test, gboost_search_pred)
    gboost_search_accuracy = r2_score(y_test, gboost_search_pred)

    print('Gradient Boosting with search Root Mean Squared Error: {}'.format(np.sqrt(gboost_search_mse)))
    print('Gradient Boosting with search Overall model accuracy: {}\n'.format(gboost_search_accuracy))
    # Accuracy is about 0.73

    ann = MLPRegressor(hidden_layer_sizes=(500, 100), max_iter=1200)
    ann.fit(X_train, y_train)
    ann_pred = ann.predict(X_test)
    ann_score = ann.score(X_test, y_test)
    ann_mse = mean_squared_error(y_test, ann_pred)
    ann_accuracy = r2_score(y_test, ann_pred)

    print('ANN Root Mean Squared Error: {}'.format(np.sqrt(ann_mse)))
    print('ANN Overall model accuracy: {}\n'.format(ann_accuracy))
    # Accuracy is about 0.75
    """
    So several methods have been tried, we can say by brute force. The accuracy
    is about 0.76. The average result, but for the first approximation it will
    do fine.
    """

if __name__ == '__main__':
    main()
