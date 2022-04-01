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

    lin_model = LinearRegression()
    lin_model.fit(X_train,y_train)   # fit the model

    print('Score on train data: {}\n'.format(lin_model.score(X_train,y_train)))
    print('Score on test data: {}'.format(lin_model.score(X_test,y_test)))

    prediction = lin_model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    accuracy = r2_score(y_test, prediction)

    print('Mean Squared Error: {}\n'.format(mse))
    print('Overall model accuracy: {}'.format(accuracy))
    print("Accuracy is about 0.6, very poor, let's try another model")
    #input("Press Enter to continue...")

    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, y_train)
    print('Score on train data: {}\n'.format(decision_tree.score(X_train,
                                                                     y_train)))
    print('Score on test data: {}\n'.format(decision_tree.score(X_test,
                                                                      y_test)))

    tree_pred = decision_tree.predict(X_test)
    tree_mse = mean_squared_error(y_test, tree_pred)
    tree_accuracy = r2_score(y_test, tree_pred)

    print('Root Mean Squared Error: {}\n'.format(np.sqrt(tree_mse)))
    print('Overall model accuracy: {}'.format(tree_accuracy))

    forest = RandomForestRegressor(n_estimators=100,
                                   max_depth=7,
                                   max_features='auto',
                                   min_samples_split=7,
                                   min_samples_leaf=3)

    forest.fit(X_train, y_train)

    print('Score on train data: {}\n'.format(forest.score(X_train, y_train)))
    print('Score on test data: {}\n'.format(forest.score(X_test, y_test)))

    forest_pred = forest.predict(X_test)
    forest_mse = mean_squared_error(y_test, forest_pred)
    forest_accuracy = r2_score(y_test, forest_pred)

    print('Root Mean Squared Error: {}\n'.format(np.sqrt(forest_mse)))
    print('Overall model accuracy: {}'.format(forest_accuracy))

    grad_boost = GradientBoostingRegressor(n_estimators=100,
                                           max_depth=7,
                                           max_features='auto',
                                           min_samples_split=7,
                                           min_samples_leaf=3,
                                           learning_rate=0.1)

    grad_boost.fit(X_train, y_train)

    print('Score on train data: {}\n'.format(grad_boost.score(X_train,
                                                                     y_train)))
    print('Score on test data: {}\n'.format(grad_boost.score(X_test,
                                                                      y_test)))

    gboost_pred = grad_boost.predict(X_test)
    gboost_mse = mean_squared_error(y_test, gboost_pred)
    gboost_accuracy = r2_score(y_test, gboost_pred)

    print('Root Mean Squared Error: {}\n'.format(np.sqrt(gboost_mse)))
    print('Overall model accuracy: {}'.format(gboost_accuracy))

    params = {'max_depth':[3,4,5,6,7],
              'max_features':['auto','sqrt','log2'],
              'min_samples_split':[2,3,4,5,6,7,8,9,10],
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}
    params['learning_rate'] = np.linspace(0.1,1,10)

    gradient_boosting = GradientBoostingRegressor()

    gboost_search = RandomizedSearchCV(gradient_boosting, params, n_jobs=-1,
                                       cv=5, verbose=2)
    gboost_search.fit(X_train, y_train)


    print('Score on train data: {}\n'.format(gboost_search.score(X_train,
                                                                     y_train)))
    print('Score on test data: {}\n'.format(gboost_search.score(X_test,
                                                                      y_test)))

    gboost_search_pred = gboost_search.predict(X_test)
    gboost_search_mse = mean_squared_error(y_test, gboost_search_pred)
    gboost_search_accuracy = r2_score(y_test, gboost_search_pred)

    print('Root Mean Squared Error: {}\n'.format(np.sqrt(gboost_search_mse)))
    print('Overall model accuracy: {}'.format(gboost_search_accuracy))

if __name__ == '__main__':
    main()
