# -*- coding: utf-8 -*-

'''This module allows to configure the application behavior'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Tuple

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def corr_matrix_display(corrmatrix: pd.DataFrame, method='Pearson', figsize: Tuple=(15, 15))-> None :
    #---------------------------------------------------------------------------
    # Build mask for lower symetric part
    #---------------------------------------------------------------------------
    mask = np.zeros_like(corrmatrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    #---------------------------------------------------------------------------
    # Build matrix color map
    #---------------------------------------------------------------------------
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    #---------------------------------------------------------------------------
    # Display result
    #---------------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix,\
                mask=mask,\
                cmap=cmap,\
                annot=True,\
                fmt='.2f',\
                square=True,\
                linewidths=.5,\
                cbar_kws={"shrink": .5})    
    plt.title('Correlation matrix - {}'.format(method))
    plt.show()
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def decorator_logger(logger_):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger_.info("Decorator invoked by function={}".format(func.__name__))
            func(*args, **kwargs)
        return wrapper
    return decorator
#-------------------------------------------------------------------------------
