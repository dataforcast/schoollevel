# -*- coding: utf-8 -*-

'''This module allows to sale data'''
import os
import pandas as pd
from sklearn.preprocessing import *
import inspect
#import logger

from util.common import Common


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class DataLoader(Common) :
    LIST_EXTENSIONS = ['csv']
    '''This class implements the data loading process that includes :
    * data loaded check integrity
    * first data processing to split dataframe as identity and feature dataframes'''
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def __init__(self, dataSource = str()):
        super().__init__()
        self._toBeHalted = True
        self._dataSource = dataSource
        self._isClean = False
        self._df = self.loadData()
        if True == self._toBeHalted :
            os._exit(-1)
        self.logger.info("Loaded dataframe shape= {}".format(self.df.shape))
    #--------------------------------------------------------------------------    

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def loadData(self) -> pd.DataFrame :
        '''This method encapsulate the detailed implementation of way 
        data is loaded.
        This forms an abstract layer between data storage and application.
        
        Current implementation supports CSV file loading only.
        '''
        if self._dataSource is not None and 0 < len(self._dataSource): 
            extension = self._dataSource.split('.')[-1]
            self.logger.debug("Extension = {}".format(extension))
            if  extension in self.LIST_EXTENSIONS :
                if 'csv' == extension :
                    return self._load_dataCSVFile()
                else :
                    pass
            else :
                self.logger\
                .fatal("Loading type not supported: {} / Program halt!"\
                .format(extension))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def _load_dataCSVFile(self) -> pd.DataFrame :
        ''' Load data from file.
        When loading file succeeded, then flag self._toBeHalted is fixed to False.
        
            OUTPUT : dataframe.
        '''
        try :
            df = pd.read_csv(self._dataSource)
            self._toBeHalted = False
        except Exception as exception :
            self.halt('{} {}'.format(exception, "Program halt!"))
        return df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def check_if_clean(self, df:pd.DataFrame) -> bool :
        '''Check wether or not dataframe has no null values.
        If dataframe has no null values, then attribute _isClean is 
        fixed to True.
        Otherwise, attribute _isClean is False leading to additional 
        processing in order to clean dataset.
        INPUT :
            * df : dataframe to be verified.
        '''
        if 1 < len(df.isna().any().unique()) :
            logger.critic("Dataframe is not clean!")
        else :
            self.logger.info('Clean dataframe')
            self._isClean = True
    #--------------------------------------------------------------------------    
    
    #--------------------------------------------------------------------------
    # Propreties
    #--------------------------------------------------------------------------
    @property
    def isClean(self):
        return self._isClean
    @isClean.setter
    def isClean(self, isClean):
        self.logger.error("Forbidden operation !")
        
    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, df_):
        self.logger.error("Forbidden operation !")
#-------------------------------------------------------------------------------
    
    
#-------------------------------------------------------------------------------            
