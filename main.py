#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
here is a simple main() module -- to demonstrate setuptools entrypoints
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import pandas as pd
import yfinance as yf  # import financial data
from datetime import datetime
from datetime import timedelta

from pandas.core.frame import DataFrame

from regression_stocks import regression_data_mod
from regression_stocks import regression_compute_mod
from regression_stocks import reg_utils

help = """
regression script
"""


def main():
    """
    startup function for running a regression as a script
    """

    try:
        ticker = sys.argv[1]
    except IndexError:
        print("you need to pass in a ticker name to process")
        print(help)
        sys.exit()

    ### data already exists
    # Specify path
    path = './data/{}.csv'.format(ticker)
    
    # Check whether the specified
    # path exists or not
    isExist = os.path.exists(path)
    print(isExist)

    ### if exist: only get missing data
    if(isExist):
        # CSV to dataframe
        df = pd.read_csv(path, index_col='Date')
        # df = df.drop(df.columns[[0]],axis=1)
        
        # Check last date from CSV
        # dateLast = df.iloc[-1]["Date"].split(' ', 1)[0] # string
        dateLast = df.index[-1].split(' ', 1)[0] # string

        # Add one day to the string
        datetimeLast = datetime.strptime(dateLast, "%Y-%m-%d").date() # dateTime
        dateNext = datetimeLast + timedelta(days=1)

        dateNextStr = dateNext.strftime("%Y-%m-%d") # string

        # Check if data missing
        if(dateNext < datetime.now().date()):
            print('Need to update data')

            # Retrieve missing data from API
            dfn = yf.download(ticker, dateNextStr)

            # Concate dataframe
            concateData = pd.concat([df, dfn])

            # Store data in csv file
            concateData.to_csv(path)
        
        else:
            concateData = df

        concateData.index = pd.to_datetime(concateData.index, utc=True) # Convert index to dateTime
        timestamps_array = regression_data_mod.reg_data_do_time_array(concateData) # Convert index to timestamp
      

        # # TEST 
        # pd.set_option('display.max_rows', None)
        # path1='./data/^GSPC.csv'
        # df1 = pd.read_csv(path1)
        # path2='./data/^GSPCNew.csv'
        # df2 = pd.read_csv(path2)

        # concate = pd.concat([df1, df2], ignore_index=True)
        # print(concate)
        # concate.to_csv('./data/test.csv')
        # reg_utils.list_columns(concate)  # DEBUG
        # concate['Date']= pd.to_datetime(concate['Date'], utc=True)
        # concate.index = concate['Date']

        # # Change index from date to timestamp
        # timestamps_array = regression_data_mod.reg_data_do_time_array(concate)

        # Get stocks values
        stock_values = regression_data_mod.reg_data_get_stocks_values(concateData)

        # Get log stocks values
        stock_values_log = regression_data_mod.reg_data_get_stocks_log(
            stock_values)
        reg_utils.list_columns(stock_values_log)  # DEBUG

        # Re-shape timestamps for linear regression
        x = regression_compute_mod.reg_comp_shape_data(timestamps_array)
        y = stock_values_log

        # Compute predicted model
        y_pred = regression_compute_mod.reg_comp_pred_model(x, y)

        # Compute SCR
        scr = regression_compute_mod.reg_comp_scr(y, y_pred)

        # Compute SCT
        sct = regression_compute_mod.reg_comp_sct(y)

        # Compute standard error of the estimate
        see = regression_compute_mod.reg_comp_sigma_estimated_non_bias(scr, y)

        # Compute standard error value
        y_pred_exp_minus2EC = regression_compute_mod.reg_comp_minus_2_sigma_values(
            y_pred, see)
        y_pred_exp_minus1EC = regression_compute_mod.reg_comp_minus_1_sigma_values(
            y_pred, see)
        y_pred_exp_plus1EC = regression_compute_mod.reg_comp_1_sigma_values(
            y_pred, see)
        y_pred_exp_plus2EC = regression_compute_mod.reg_comp_2_sigma_values(
            y_pred, see)

        # Plot regression
        fig = plt.figure(facecolor='yellow')
        ax = fig.add_subplot(1, 1, 1)

        x2=concateData.index.values

        ax.plot(x2, stock_values, color='blue')  # nuage de point
        ax.plot(x2, np.exp(y_pred), color='red')  # nuage de point
        ax.plot(x2, y_pred_exp_minus2EC, color='purple')  # nuage de point
        ax.plot(x2, y_pred_exp_minus1EC, color='purple')  # nuage de point
        ax.plot(x2, y_pred_exp_plus1EC, color='purple')  # nuage de point
        ax.plot(x2, y_pred_exp_plus2EC, color='purple')  # nuage de point

        plt.show()

        sys.exit()


        
        

    else:
        # Get all data from API
        dfn = yf.download(ticker, dateLast)
        dfn.to_csv(path)

    
    '''
    try:
        date_start = sys.argv[2]
    except IndexError:
        print("you need to pass in a start date as AAAA-MM-DD to process")
        print(help)
        sys.exit()

    try:
        date_end = sys.argv[3]
    except IndexError:
        print("you need to pass in a start date as AAAA-MM-DD to process")
        print(help)
        sys.exit()
    '''

    # do the real work:
    print("Getting data: %s from %s to %s" % (ticker, date_start, date_end))

    # Get data as pandas dataframe
    data = regression_data_mod.reg_data_import_data_for_ticker(
        ticker, date_start, date_end)
    reg_utils.list_columns(data)  # DEBUG

    # Change index from date to timestamp
    timestamps_array = regression_data_mod.reg_data_do_time_array(data)

    # Get stocks values
    stock_values = regression_data_mod.reg_data_get_stocks_values(data)

    # Get log stocks values
    stock_values_log = regression_data_mod.reg_data_get_stocks_log(
        stock_values)
    reg_utils.list_columns(stock_values_log)  # DEBUG

    # Re-shape timestamps for linear regression
    x = regression_compute_mod.reg_comp_shape_data(timestamps_array)
    y = stock_values_log

    # Compute predicted model
    y_pred = regression_compute_mod.reg_comp_pred_model(x, y)

    # Compute SCR
    scr = regression_compute_mod.reg_comp_scr(y, y_pred)

    # Compute SCT
    sct = regression_compute_mod.reg_comp_sct(y)

    # Compute standard error of the estimate
    see = regression_compute_mod.reg_comp_sigma_estimated_non_bias(scr, y)

    # Compute standard error value
    y_pred_exp_minus2EC = regression_compute_mod.reg_comp_minus_2_sigma_values(
        y_pred, see)
    y_pred_exp_minus1EC = regression_compute_mod.reg_comp_minus_1_sigma_values(
        y_pred, see)
    y_pred_exp_plus1EC = regression_compute_mod.reg_comp_1_sigma_values(
        y_pred, see)
    y_pred_exp_plus2EC = regression_compute_mod.reg_comp_2_sigma_values(
        y_pred, see)

    # Plot regression
    fig = plt.figure(facecolor='yellow')
    ax = fig.add_subplot(1, 1, 1)

    x2=data.index.values

    ax.plot(x2, stock_values, color='blue')  # nuage de point
    ax.plot(x2, np.exp(y_pred), color='red')  # nuage de point
    ax.plot(x2, y_pred_exp_minus2EC, color='purple')  # nuage de point
    ax.plot(x2, y_pred_exp_minus1EC, color='purple')  # nuage de point
    ax.plot(x2, y_pred_exp_plus1EC, color='purple')  # nuage de point
    ax.plot(x2, y_pred_exp_plus2EC, color='purple')  # nuage de point

    plt.show()

main()