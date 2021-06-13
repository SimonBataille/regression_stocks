#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
