#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
here is a simple main() module -- to demonstrate setuptools entrypoints
"""

import sys
import os

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
    regression_data_mod.reg_comp_shape_data(timestamps_array)


