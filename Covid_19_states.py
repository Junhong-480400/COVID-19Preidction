# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 07:44:13 2020

@author: PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import matplotlib

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def CookDataFrame(df, state_name, lag=5):
    """

    Parameters
    ----------
    df : Pandas Dataframe
        The original data including training set and test set
    state_name : Python String
        The targeted state
    lag : Python Int
        The time lag, a hyperparameter

    Returns
    -------
    x : 2D Numpy Array 
        Predictors
    y : 2D Numpy Array
        Responses

    """
    # select dataframe whose state are state_name 
    df = df[df['state'] == state_name]
    
    # choose predictors(no abnormal nan) and response
    x_features = ['negative']
    y_features = 'positive'
    
    x = df[x_features].fillna(0).values
    y = df[y_features].fillna(0).values
    
    # fillna another time
    if x.shape[0] != 181:
        x = np.concatenate((np.zeros((181 - x.shape[0], x.shape[1])), x), axis = 0)
        y = np.concatenate((np.zeros((181 - y.shape[0])), y) ,axis = 0)
    
    # gain information from the lag
    N_days = x.shape[0]
    lag_matrix = np.empty((N_days, lag))
    
    # form the lagged matrix
    for i in range(lag):
        lag_matrix[i] = np.zeros((lag, ))
    for i in range(N_days-lag):
        lag_matrix[i+lag] = y[i:i+lag].reshape(-1)
        
    # change predictors and response as lagged matrix emerges
    x = np.concatenate((x, lag_matrix), axis=1)
    y = y.reshape(-1, 1)
    
    return x, y

def PolynomialRegMinDeg(df, itrain, itest, state_name, n_folds=4, lag=5):
    """
    
    Parameters
    ----------
    df : Pandas Dataframe
        The original data including training set and test set
    state_name : Python String
        The targeted state
    n_folds : Python Int
        The number of folds needed to be in cross validation part
    lag : Python Int
        The time lag, a hyperparameter

    Returns
    -------
    mse : 1D Numpy Array shaped
        Mean_squared_errors of every degree
    mindeg : Python int
        The degree(index) which minimize the mse list
        
    """
    # use function to get cooked predictors denoted x and responses denoted y
    x, y = CookDataFrame(df, state_name, lag)
    
    # train_test_split
    train_all_x = x[itrain, :]
    
    train_all_y = y[itrain, :]
    
   # set degrees as a numpy array
    degrees = np.arange(21)
    
    # initialize the mse array
    mse = np.empty((len(degrees)))
    
    for j in degrees:
        for i in range(train_all_x.shape[1]):
            if i == 0:
                train_all_X = PolynomialFeatures(j).fit_transform(train_all_x[:,i].reshape(-1,1))
            else:
                train_all_X = np.concatenate((train_all_X, PolynomialFeatures(j).fit_transform(train_all_x[:,i].reshape(-1,1))), axis=1)
    
        model = LinearRegression()
    
        # the average value of scores after cross validation
        mse[j] = np.mean(cross_val_score(model, train_all_X, train_all_y, cv=n_folds, scoring='neg_mean_squared_error'))
    
    
    mse = np.abs(mse)
    
    #choose the least of our mse array
    mindeg = np.argmin(mse)
    
    return mse, mindeg

def PolynomialRegMinLag(df, itrain, itest, state_name, n_folds=4, mindeg=3):
    """

    Parameters
    ----------
    df : Pandas Dataframe
        The original data including training set and test set
    state_name : Python String
        The targeted state
    n_folds : Python Int
        The number of folds needed to be in cross validation part
    times : Python Int
        The number of times to operate bootstraps(sampling aka)
    mindeg : Python Int
        The degree(index) which minimize the mse list

    Returns
    -------
    -- : Python Int
        Best performed lag to minimize mse list.

    """
    lags = np.arange(2, 14)
    
    mse_mean_mindeg = np.empty((len(lags), ))
    mindeg_lag = np.empty((len(lags), ))
    
    for i in range(len(lags)):
        
        mse_mean, mindeg = PolynomialRegMinDeg(df, itrain, itest, state_name, lags[i], n_folds)
        
        mse_mean_mindeg[i] = mse_mean[mindeg]
        mindeg_lag[i] = mindeg
        
    return mse_mean_mindeg, mindeg_lag, int(lags[np.argmin(mse_mean_mindeg)]), int(mindeg_lag[np.argmin(mse_mean_mindeg)])

def PolynomialRegPredict(df, itrain, itest, state_name, mindeg=3, minlag=5):
    # use function to get cooked predictors denoted x and responses denoted y
    x, y = CookDataFrame(df, state_name, minlag)
    
    # train_test_split
    train_all_x = x[itrain, :]
    test_all_x = x[itest, :]
    
    train_all_y = y[itrain, :]
    test_all_y = y[itest, :]
    
    # produce set in some mindeg
    for i in range(x.shape[1]):
        if i == 0:
            train_all_X = PolynomialFeatures(mindeg).fit_transform(train_all_x[:, i].reshape(-1, 1))
            test_all_X = PolynomialFeatures(mindeg).fit_transform(test_all_x[:, i].reshape(-1, 1))
        else:
            train_all_X = np.concatenate((train_all_X, PolynomialFeatures(mindeg).fit_transform(train_all_x[:, i].reshape(-1, 1))), axis=1)
            test_all_X = np.concatenate((test_all_X, PolynomialFeatures(mindeg).fit_transform(test_all_x[:, i].reshape(-1, 1))), axis=1)
    
    model = LinearRegression()
    model.fit(train_all_X, train_all_y)
    
    predict_y_test = model.predict(test_all_X)
    r2_test = r2_score(test_all_y, predict_y_test)
    
    return  itest, test_all_y, predict_y_test, r2_test, model, mindeg, minlag

def Operations(df, state_name):
    # train_test_split
    itrain, itest = train_test_split(np.arange(181), test_size = 0.1, random_state=0)
    
    # use function to get cooked predictors denoted x and responses denoted y    
    mse_mean_mindeg, mindeg_lag, minlag, mindeg = PolynomialRegMinLag(df, itrain, itest, state_name)
    
    itest, test_all_y, predict_y_test, r2_test, model, mindeg, minlag = PolynomialRegPredict(df, itrain, itest, state_name, mindeg, minlag)
    
    return itest, test_all_y, predict_y_test, r2_test, model, mindeg, minlag, mse_mean_mindeg, mindeg_lag

def PlotPredict(df, state_name_modeling, state_name_predicting, class_name, my_figsize=(10, 6), option=0):
    """
    
    Parameters
    ----------
    df : Pandas Dataframe
        The original data including training set and test set
    state_name_modeling : Python String
        The targeted state for modeling
    state_name_predicting : Python String
        The targeted state for predicting
    class_name : Python Int
        The targeted class
    my_figsize : Python Tuple
        Set the size of my figure
    option : Python Int
        The option of title
        
    Returns
    -------
    NONE
        
    """
    itest, test_all_y, predict_y_test, r2_test, model, mindeg, minlag, mse_mean_mindeg, mindeg_lag = Operations(df, state_name_modeling)
    
    if option == 0:
        plt.figure(figsize=my_figsize)
        color_lines = sns.color_palette()
        color_points = ['bo', 'go', 'ko', 'yo', 'mo']
    
        for j, state in enumerate(state_name_predicting):
            if state != state_name_modeling:
                x, y = CookDataFrame(df, state, lag=minlag)
                
                for i in range(x.shape[1]):
                    if i == 0:
                        X = PolynomialFeatures(mindeg).fit_transform(x[:, i].reshape(-1, 1))
                    else:
                        X = np.concatenate((X, PolynomialFeatures(mindeg).fit_transform(x[:, i].reshape(-1, 1))), axis=1)
                
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                plt.plot(days, y_pred, color=color_lines[j%len(color_lines)], label=state+' Pure Predictions and R2:{:.5f}'.format(r2))
                plt.plot(days, y, color_points[j%len(color_points)], markersize=3, alpha=0.4, label=state+' Pure Ground Truth')
                plt.title('Class '+str(class_name)+' Predictions')
           
            else:
                plt.plot(np.sort(days[itest]), np.sort(predict_y_test, axis=0), color=color_lines[j%len(color_lines)], label=state+' Test Predictions and R2:{:.5f}'.format(r2_test))
                plt.plot(np.sort(days[itest]), np.sort(test_all_y, axis=0), color_points[j%len(color_points)], markersize=3, alpha=0.4, label=state+' Ground Truth')
                plt.title(state+' Self Test Predictions')
            
        plt.xlabel('Days')
        plt.ylabel('Positive')
        plt.grid()
        plt.legend()
        
    else:
        color_lines_points = ['bo-', 'go-', 'ko-', 'yo-', 'mo-']
        color_axvlines = sns.color_palette()
        
        plt.figure(figsize=my_figsize)
        lags = np.arange(2, 14)
        plt.plot(lags, mse_mean_mindeg, color_lines_points[0])
        plt.axvline(minlag, 0, 1, color=color_axvlines[0]),
        plt.title(state_name_modeling+' MSE of different lags in its best degree')
        plt.xlabel('Lags')
        plt.ylabel('MSE')
        plt.grid()
        
        plt.figure(figsize=my_figsize)
        lags = np.arange(2, 14)
        plt.plot(lags, mindeg_lag, color_lines_points[1])
        plt.axvline(minlag, 0, 1, color=color_axvlines[1])
        plt.title(state_name_modeling+' best degrees of different lags')
        plt.xlabel('Lags')
        plt.ylabel('Degrees')
        plt.grid()
        
    
    return 

df = pd.read_csv('F:\\us_states_daily.csv')
df = df.iloc[::-1] # get reverse to plot
days = np.arange(181) # would be used in too many functions

states = np.unique(df.state.values)
class_1 = ['AS']
class_2 = ['MP', 'HI', 'AK', 'VT', 'GU', 'MT', 'WV', 'ME']
class_3 = ['NM', 'ND', 'NH', 'OR', 'VI', 'KY', 'WY', 'OK', 'MN', 'MI', 'WA', 'MO', 'WI', 'PR', 'CA', 'OH', 'TN', 'NC', 'IL', 'CT', 'UT', 'DC', 'AR', 'NY']
class_4 = ['SD', 'VA', 'LA', 'DE', 'KS', 'IN', 'CO', 'NV', 'IA', 'NE', 'RI', 'PA', 'NJ', 'ID', 'MD', 'MA', 'TX', 'MS', 'GA', 'AL', 'FL', 'SC']
class_5 = ['AZ']

print(pd.__version__)
print(sklearn.__version__)
print(matplotlib.__version__)
print(sns.__version__)
print(np.__version__)

# PlotPredict(df, 'AS', class_1, 1)

# PlotPredict(df, 'AK', class_2[:2], 2)
# PlotPredict(df, 'AK', ['AK'], 2)

# PlotPredict(df, 'CA', class_3[:5], 3)
# PlotPredict(df, 'CA', ['CA'], 3)

# PlotPredict(df, 'TX', class_4[:5], 4)
# PlotPredict(df, 'TX', ['TX'], 4)

# PlotPredict(df, 'AZ', class_5, 5)

# PlotPredict(df, 'AS', class_1, 1, option=1)
# PlotPredict(df, 'AK', ['AK'], 2, option=1)
# PlotPredict(df, 'CA', ['CA'], 3, option=1)
# PlotPredict(df, 'TX', ['TX'], 4, option=1)
# PlotPredict(df, 'AZ', class_5, 5, option=1)
    
    