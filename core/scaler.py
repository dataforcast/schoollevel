# -*- coding: utf-8 -*-

'''This module contains utilities to scale data'''

import pandas as pd
import numpy as np
import inspect

from util.common import Common

from sklearn.preprocessing import *
from sklearn.preprocessing import FunctionTransformer

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class IdentityScaler(Common):
    '''Returns identical values for incoming data.'''
    def __init__(self):
        super().__init__()
        self._transformer = FunctionTransformer(func=lambda x: x\
        , inverse_func=lambda x: x)
        
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self._transformer.fit(X.values)
        else:
            self._transformer.fit(X)
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return self._transformer.transform(X.values)
        else:
            return self._transformer.transform(X)
    def fit_transform(self, X) :
        if isinstance(X, pd.DataFrame):
            return self._transformer.fit_transform(X.values)
        else:
            return self._transformer.fit_transform(X)
    def inverse_transform(self, X):
        return X
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class MyScaler(Common):
    '''This class is a wrapper for all scaler classes.
    It allows to apply same functions as original scaler 
    and to return data structured as dataframe.
    '''
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def __init__(self, scaler, list_feature, logger=None) :
        super().__init__(logger=logger)
        self._scaler = scaler
        self.list_feature = list_feature.copy()
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def _get_list_other_feature(self, df:pd.DataFrame) :
        return [other_feature for other_feature in df.columns \
                if other_feature not in self.list_feature]
    #--------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def _scale_function(self, df:pd.DataFrame, scale_function) -> pd.DataFrame:
        '''Generic function for scaler functions fit_transform or transform
        INPUT
            * df : dataframe with some features to be scaled; those features 
            are in list self.list_feature
            * scale_function : either fit_transform or transform
        OUTPUT
            Scaled dataframe
        '''
        df_clean = df[self.list_feature].transpose().drop_duplicates().transpose()
        df_scaled = pd.DataFrame(data= scale_function(df_clean)\
                                , index = df_clean.index\
                                , columns=df_clean.columns)
        #---------------------------------------------------------------------
        # Get other features that have not been scaled
        #---------------------------------------------------------------------
        list_other = self._get_list_other_feature(df)

        #---------------------------------------------------------------------
        # Concatenate scaled and non scaled features to return a dataframe 
        # with sae number of features then the original one.
        #---------------------------------------------------------------------
        if 0 < len(list_other):
            df_scaled =  pd.concat((df_scaled, df[list_other]), axis=1)
        else:
            pass
        return df_scaled             
    #--------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def fit_transform(self, df:pd.DataFrame) :
        return self._scale_function(df, self._scaler.fit_transform)                  
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def fit(self, df:pd.DataFrame) :
        df_clean = df[self.list_feature].transpose().drop_duplicates().transpose()
        self._scaler.fit(df_clean)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self._scale_function(df, self._scaler.transform)                  
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def inverse_transform(self, df:pd.DataFrame) :
        return pd.DataFrame(data = self._scaler.inverse_transform(df[self.list_feature])\
                            , index = df.index, columns = self.list_feature)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Propreties
    #--------------------------------------------------------------------------
    @property
    def scaler(self):
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))
        return self._scaler
    @scaler.setter
    def scaler(self, scaler):
        self.logger.error("Forbidden operation !")

    @property
    def list_other_feature(self, df):
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))
        return [other_feature for other_feature in df.columns \
                if other_feature not in self.list_feature]
    @list_other_feature.setter
    def list_other_feature(self, df):
        self.logger.error("{} : Forbidden operation !".format(inspect.currentframe().f_code.co_name))
        
#-------------------------------------------------------------------------------            
