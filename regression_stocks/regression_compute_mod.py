#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression  # for regression linear
import numpy as np  # to handle nparray data
import math

"""
y = stock_log = log (stock)
liear(x,y)
y_pred = slope*x + intercept

exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
"""



def reg_comp_shape_data(timestamps):
    """
    # this array is required to be two-dimensional, or to be more precise, to have one column and as many rows as necessary
    # x = timestamps.reshape(-1, 1)
    """

    """
    >>> timestamps_array.shape 
    (5394,)
    >>> timestamps_array.reshape(-1, 1)
    array([[730122],
           [730123],
           [730124],
           ...,
           [737949],
           [737950],
           [737951]])
    >>> timestamps_array.reshape(-1, 1).shape
    (5394, 1)
    >>> timestamps_array
    array([730122, 730123, 730124, ..., 737949, 737950, 737951])
    >>> timestamps_array.shape
    (5394,)
    >>> timestamps_array.reshape(-1, 1)
    array([[730122],
           [730123],
           [730124],
           ...,
           [737949],
           [737950],
           [737951]])
    >>> timestamps_array.reshape(-1, 1).shape
    (5394, 1)
    """

    return timestamps.reshape(-1, 1)


def reg_comp_pred_model(x, y):
    """
    Compute predicted model
    """
    model = LinearRegression()

    model.fit(x, y)
    r_sq = model.score(x, y)  # R²
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')

    return y_pred


def reg_comp_scr(y, y_pred):
    """
    SS_Residual = sum((y-y_pred)**2) : SCR somme des carrés résiduels
    """
    return sum((y-y_pred)**2)


def reg_comp_sct(y):
    """
    SS_Total = sum((y-np.mean(y))**2) : SCT somme des carrés totaux
    """
    return sum((y-np.mean(y))**2)


def reg_comp_sigma_estimated_non_bias(SS_Residual, y):
    """
    sigma_estimated_non_bias = SS_Residual/(len(y)-2)
    """
    return SS_Residual/(len(y)-2)


def reg_comp_regression_info(SS_Residual, SS_Total, sigma_estimated_non_bias, y, x):
    """
    Compute info from regression
    """
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)

    print('TEST : ', r_squared, adjusted_r_squared, sigma_estimated_non_bias)
    print('$\sigma$', sigma_estimated_non_bias, math.sqrt(sigma_estimated_non_bias))


"""
def reg_comp_standard_error_values(method, y_pred, sigma_estimated_non_bias):
       match(method):
           case("-2"):
              return np.exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
           case("-1"):
              return np.exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
           case("1"):
              return np.exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
           case("2"):
              return np.exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
           case _:
            raise ValueError("Invalid arguments")
"""


def reg_comp_2_sigma_values(y_pred, sigma_estimated_non_bias):
    """
    Return 
    """
    return np.exp(y_pred)*(1+2*math.sqrt(sigma_estimated_non_bias))


def reg_comp_1_sigma_values(y_pred, sigma_estimated_non_bias):
    """
    Return 
    """
    return np.exp(y_pred)*(1+math.sqrt(sigma_estimated_non_bias))


def reg_comp_minus_1_sigma_values(y_pred, sigma_estimated_non_bias):
    """
    Return 
    """
    return np.exp(y_pred)*(1-math.sqrt(sigma_estimated_non_bias))


def reg_comp_minus_2_sigma_values(y_pred, sigma_estimated_non_bias):
    """
    Return 
    """
    return np.exp(y_pred)*(1-2*math.sqrt(sigma_estimated_non_bias))
