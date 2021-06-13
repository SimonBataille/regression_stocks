#!/usr/bin/python3
# -*- coding: utf-8 -*-


import yfinance as yf  # import financial data
import datetime as dt
import pandas as pd  # to read pandas data from yfinance
import numpy as np # to handle nparray data

# import from regression module
from regression_stocks import reg_utils


def reg_data_import_data_for_ticker(ticker, date_start, date_end):
    """
    Import data from yfinance as pandas DataFrame
    """
    return yf.download(ticker, date_start, date_end)


def reg_data_do_time_array(data):
    """
    Return final numpy timestamps array

    >>> data.index
    DatetimeIndex(['2000-01-03', '2000-01-04', '2000-01-05', '2000-01-06',
               '2000-01-07', '2000-01-10', '2000-01-11', '2000-01-12',
               '2000-01-13', '2000-01-14',
               ...
               '2021-05-27', '2021-05-28', '2021-06-01', '2021-06-02',
               '2021-06-03', '2021-06-04', '2021-06-07', '2021-06-08',
               '2021-06-09', '2021-06-10'],
              dtype='datetime64[ns]', name='Date', length=5394, freq=None)

    >>> type(data.index)
    <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

    >>> data.index.shape
    (5394,)


    >>> data.index.map(dt.datetime.toordinal)
    Int64Index([730122, 730123, 730124, 730125, 730126, 730129, 730130, 730131,
            730132, 730133,
            ...
            737937, 737938, 737942, 737943, 737944, 737945, 737948, 737949,
            737950, 737951],
           dtype='int64', name='Date', length=5394)

    >>> type(data.index.map(dt.datetime.toordinal))
    <class 'pandas.core.indexes.numeric.Int64Index'>

    >>> data.index.map(dt.datetime.toordinal).values
    array([730122, 730123, 730124, ..., 737949, 737950, 737951])

    >>> type(data.index.map(dt.datetime.toordinal).values)
    <class 'numpy.ndarray'>

    >>> data.index.map(dt.datetime.toordinal).values.shape
    (5394,)
    """

    # timestamps are seconds from 1970-01-01-0-0-0
    # data_index_map = data.index.map(dt.datetime.toordinal) # convert data.index from datetimes to timestamps
    # timestamps_array = data_index_map.values # get a numpy array

    return data.index.map(dt.datetime.toordinal).values


def reg_data_get_stocks_values(data):
    """
    Store Adj Close values in numpy array

    >>> data['Adj Close']
    Date
    2000-01-03    12.473144
    2000-01-04    12.109828
    2000-01-05    12.109828
    2000-01-06    11.840109
    2000-01-07    11.805033
                    ...    
    2021-06-04    48.570000
    2021-06-07    48.189999
    2021-06-08    48.369999
    2021-06-09    48.590000
    2021-06-10    48.299999
    Name: Adj Close, Length: 5394, dtype: float64

    >>> type(data['Adj Close'])
    <class 'pandas.core.series.Series'>

    >>> data['Adj Close'].shape
    (5394,)

    >>> data['Adj Close'].to_numpy()
    array([12.47314358, 12.109828  , 12.109828  , ..., 48.36999893,
       48.59000015, 48.29999924])

    >>> data['Adj Close'].to_numpy().shape
    (5394,)

    >>> type(data['Adj Close'].to_numpy())
    <class 'numpy.ndarray'>

    >>> type(data['Adj Close'].to_numpy())
    <class 'numpy.ndarray'>
    """
    # Get stock values Adj Close
    return data['Adj Close'].to_numpy()


def reg_data_get_stocks_log(stocks_array):
    """
    Return array filled with log of stocks values
    stocks_array is numpy.ndarray
    """
    return np.log(stocks_array)
