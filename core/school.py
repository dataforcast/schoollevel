# -*- coding: utf-8 -*-

'''This module contains utilities to scale data'''
import ydata_profiling
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import inspect
import logging

from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture

from sklearn.ensemble import IsolationForest

from sklearn.calibration import CalibratedClassifierCV

from util import common
from util.common import Common

from core.scaler import IdentityScaler, MyScaler
from util.analysis import classifier_grid_crossval


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
class School(Common):
    """This class implements functions related to schools
    dataset transformation"""

    # --------------------------------------------------------------------------
    # Class attributes
    # --------------------------------------------------------------------------
    LIST_IDENTITY = ['FirstName', 'FamilyName']
    INDEX_STUDENT = 'StudentID'
    TARGET = 'FinalGrade'
    COL_OUTLIER_NAME = 'OUTLIERS'
    #CLUSTER = 'Clusters'
    LABEL   = 'Label'
    COL_DIMENSION_NAME = "Col"

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def __init__(self, df: pd.DataFrame, html_profile="./html/schools_analysis.html"):
        super().__init__()
        self._html_profile = html_profile
        self._list_feature_qualitative = list()
        self._list_feature_quantitative = list()
        self._list_col_data = list()

        self._df_student = None
        self._df_identity = None
        self._df_feature = None
        self._df_dummy = None
        self._df_target = None
        self._df_data = None
        self._df_label = None
        self._df_data_proj = None
        self._oLabelEncoder = None
        self._oOneHotEncoder = None
        self._oScaler = None
        self._estimator = 'regressor'
        self._oBestClassifier = None
        self._oBestClassifierCalibrated = None
        self._dict_df_X_train_test = dict()
        self.nbgroup = 0
        self._is_calibrated = False
        self._list_feature_remove = list()

        self._store_df(df)
        self.df_build_identity_and_feature()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _store_df(self, df: pd.DataFrame) -> None:
        """Rebuild dataframe index and check for rows duplication.
        """
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))
        # TBD : check wether or not self.INDEX_STUDENT is in th columns names
        if School.INDEX_STUDENT not in df.columns:
            message = "_store_df : no column name= {} in dataframe columns for use as index!" \
                .format(self.INDEX_STUDENT)
            # self.halt(message)
            # os._exit(-1)
        else:
            self._df_student = df.reset_index().drop(columns=[self.INDEX_STUDENT], axis=1, inplace=False)
            self._df_student.drop(columns=['index'], axis=1, inplace=True)
            self._df_student.index.name = self.INDEX_STUDENT
            has_duplicated_rows = True in self._df_student.duplicated().unique()
            if has_duplicated_rows:
                self.logger.warning("Duplicated rows for dataframe")
            else:
                self.logger.info("No duplicated rows for dataframe")
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def feature_restore(self):
        self.df_data = None
        self._list_feature_quantitative = list()
        self._list_feature_qualitative = list()

        # ----------------------------------------------------------------------
        # Yield qualitative list of features
        # ----------------------------------------------------------------------
        self._yield_qualitativeFeatures()

        # ----------------------------------------------------------------------
        # Yield quantitative list of features
        # ----------------------------------------------------------------------
        self._yield_quantitativeFeatures()
        
        # ----------------------------------------------------------------------
        # List of removed features is droped
        # ----------------------------------------------------------------------
        self._list_feature_remove = list()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _yield_qualitativeFeatures(self):
        '''Yield qualitative features from feature dataframe.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        if 0 < len(self._list_feature_qualitative):
            pass
        else:
            self._list_feature_qualitative = \
                self._df_feature.select_dtypes(include=['object', 'category', 'bool']).columns

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _yield_quantitativeFeatures(self):
        '''Yield quantitative features from feature dataframe.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))
        if 0 < len(self._list_feature_quantitative):
            pass
        else:
            self._list_feature_quantitative = \
                [feature for feature in self._df_feature.columns \
                 if feature not in self._list_feature_qualitative and feature != School.TARGET]

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    # @decorator_logger(this.logger)
    def df_build_identity_and_feature(self) -> None:
        '''Split students dataframe in two dataframes :
        1) identity dataframe that contains columns related to student identity
        2) feature dataframe that contains columns related to student features
        other than identities

        Both split dataframes share the same indexes.
        
        This avoids to apply analysis over variables without any feature of
        interest.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        if (self._df_identity is not None) and (self._df_feature is not None):
            pass
        else:
            if self._df_identity is None:
                list_colIdentity = School.LIST_IDENTITY
                df = self._df_student

                # --------------------------------------------------------------
                # Check input validity
                # --------------------------------------------------------------
                if 0 < len(list_colIdentity):
                    pass
                else:
                    message = "df_build_identity_and_feature : no column for identities! "
                    self.halt(message)
                    os._exit(-1)

                # --------------------------------------------------------------
                # Create identity dataframe
                # --------------------------------------------------------------
                self._df_identity = df[list_colIdentity]
                self.logger.info("df identity shape= {}". \
                                 format(self._df_identity.shape))
            else:
                pass

            if self._df_feature is None:
                # --------------------------------------------------------------
                # Create features dataframe
                # --------------------------------------------------------------
                list_colFeature = [colFeature for colFeature in self._df_student.columns \
                                   if colFeature not in list_colIdentity]
                if 0 < len(list_colFeature):
                    self._df_feature = self._df_student[list_colFeature]
                    self.logger.info("df feature shape= {}". \
                                     format(self._df_feature.shape))
                else:
                    message = "No column for any feature! "
                    self.halt(message)
                    os._exit(-1)
            else:
                pass

        # ----------------------------------------------------------------------
        # Yield quantitative and qualitative list of features
        # ----------------------------------------------------------------------
        self.feature_restore()

    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def get_dict_label_index(self):
        '''Returns a dictionary structured as {label:index} where:
        -> label is a class identifier issued from the dataset labellisation
        -> index is the dataframe index for each label.

        When labellisation takes place, then df_label is not None. Otherwise,
        an empty dictionary is returned.
        '''
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(School.LOGGER_PROMPT.format(function_name))
        dict_label_index = dict()
        if self.df_label is None:
            self.logger.warning(School.LOGGER_PROMPT + ": {}" \
                                .format(function_name, "No dataset labellisation took place!"))
        else:
            df_y = pd.concat((self.dict_df_X_train_test['df_y_train'] \
                                  , self.dict_df_X_train_test['df_y_test']) \
                             , axis=0)
            dict_label_index = dict()
            for label in df_y[School.LABEL].unique():
                index_i = df_y[df_y[School.LABEL] == label].index
                dict_label_index[label] = index_i
        return dict_label_index

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def report(self):
        '''Use pandas-profiling to generate a report from the school dataset 
        profile and dump it into a HTML file. 
        If directory that hosts HTML file does not exists then it is created.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        if self._df_feature is None:
            message = "No dataframe for features analysis !"
            self.logger.warning(message)
            logging.warning(message)
        else:

            dir_path = common.yield_pathdir_from_pathfile(self._html_profile)

            if not Path(dir_path).is_dir():
                message = "Path directory does not exists: {}".format(dir_path)
                logging.info(message)
                self.logger.info(message)
                path = Path(dir_path)
                if not path.exists():
                    message = "Directory created: {}".format(dir_path)
                    try:
                        path.mkdir()
                        self.logger.info(message)
                        logging.info(message)
                    except Exception as exception:
                        self.logger.warning(exception)
                        logging.warning(exception)
            else:
                try:
                    self.profile = ydata_profiling.ProfileReport(self._df_feature)
                except Exception as exception:
                    self.logger.warning(exception)
                try:
                    self.profile.to_file(self._html_profile)
                    message = "Dataset profile dumped into file: {}".format(self._html_profile)
                    self.logger.info(message)
                    logging.info(message)
                except Exception as exception:
                    self.logger.warning(exception)

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _df_get_dummy_feature(self) -> pd.DataFrame:
        '''Transform qualitative features as numeric ones using 
        one hot encoding.
        During encoding, the last encoded column is droped in order 
        to avoid encoded correlations.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        if self._df_dummy is None:
            self._oOneHotEncoder = OneHotEncoder(sparse=False, drop='first')
            self._df_dummy = pd.DataFrame()
            df = self.df_feature_qual

            self._df_dummy = \
                pd.DataFrame(self._oOneHotEncoder.fit_transform(self.df_feature_qual) \
                             , columns=self._oOneHotEncoder.get_feature_names_out(
                        input_features=self.df_feature_qual.columns))
        else:
            pass
        return self._df_dummy

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def df_get_outliers_zscore(self, df: pd.DataFrame, threshold: int = 3, target='StudentID') -> pd.DataFrame:
        '''Yield outliers for each column and store them in a dataframe 
        that is returned.
        Default method is the Z-score
        INPUT:
            * df : dataframe from which outliers are extracted
            * threshold : the number of standard deviations
        OUTPUT :
            * daframe containing outliers with same index then input dattaframe.
            All columns wher all values are undefined are dropped. 
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        df_outlier = pd.DataFrame()

        for feature_quant in self.df_feature_quant.columns:
            data = self.df_feature_quant[feature_quant].values
            mean, std_dev = np.mean(data), np.std(data)
            arr_z_score = (data - mean) / std_dev

            arr_outlier = data[np.abs(arr_z_score) > threshold]
            if 0 < len(arr_outlier):
                min_outlier = np.min(arr_outlier)
                filter_outlier = self.df_feature_quant[feature_quant] >= min_outlier
                index_outlier = self.df_feature_quant[filter_outlier].index
                df_outlier = pd.concat((df_outlier \
                                            , self.df_feature_quant.loc[index_outlier][feature_quant].reset_index()) \
                                       , axis=0 \
                                       , ignore_index=True)
        df_outlier.set_index(School.TARGET, inplace=True)

        return df_outlier.dropna(axis=1, how='all', inplace=False)
        # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def dimension_reduction(self, dict_method=dict()) -> None:
        '''
        Reduce dimension of the dataset and update self._df_data with the reduced
        dataset.
        
        INPUT:
            * dimension : the resulted dimension for dataset 
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))
        df_data = self.df_data
        df_data = df_data[self.list_col_feature]
        algo = str()
        if 'algo' not in dict_method:
            algo = 'tsne'
            dict_method = {'algo': algo, 'n_components': 2, 'perplexity': 25, 'random_state': 13}
            self.logger.warning("{}: No reduction method provided! fallback with t-SNE param={}" \
                                .format(inspect.currentframe().f_code.co_name, dict_method))
        else:
            algo = dict_method['algo']

        dict_method.pop('algo')
        if 'tsne' == algo:
            try:
                reduction = dict_method['n_components']
                list_col_reduction = [School.COL_DIMENSION_NAME + str(dim) for dim in range(reduction)]
                oTSNE = TSNE(**dict_method)
                self.df_data = pd.DataFrame(data=oTSNE.fit_transform(df_data.values) \
                                            , index=df_data.index \
                                            , columns=list_col_reduction)
                self.df_data = pd.concat((self.df_data, self._df_label), axis=1)
            except Exception as exception:
                self.logger.error(inspect.currentframe().f_code.co_name + ": {}".format(exception))

        else:
            self.logger.error(inspect.currentframe().f_code.co_name + ": {} not implemented".format(algo))

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def df_get_outliers(self \
                        , contamination=0.1 \
                        , method='isolationforest' \
                        , outliername='OUTLIERS' \
                        , perplexity=20 \
                        , is_displayed=True) -> pd.DataFrame:
        '''Returns a dataframe with outliers. Outliers are detected with IsolationForest 
        algorithm by default.'''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        df_X = self.df_data
        list_feature = list(df_X.columns)
        X = df_X.values

        oIsolationForest = IsolationForest(contamination=contamination, random_state=13)
        oIsolationForest.fit(X)
        y_pred = oIsolationForest.predict(X)
        df_outlier = pd.DataFrame(data=y_pred, index=df_X.index, columns=[outliername])
        index_outlier = df_outlier[df_outlier[outliername] == -1].index

        print("Nb of outliers={}".format(len(index_outlier)))

        index_inlier = [index for index in self.df_target.index if index not in index_outlier]

        if is_displayed:
            y = df_outlier[outliername].values

            list_color = [1] * (len(index_inlier) + len(index_outlier))
            try:
                for index in index_outlier:
                    list_color[index] = 0
            except Exception as exception:
                self.logger.error("***Outlier index={} : {}".format(index, exception))

            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=13)
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=list_color, cmap='gray', edgecolors='k', s=60)

            legend1 = plt.legend(*scatter.legend_elements(), title="Outliers", loc="best")
            plt.gca().add_artist(legend1)

            plt.title("t-SNE - with perplexity= {}".format(perplexity))
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()
        df_mask = df_outlier[outliername] == -1
        return df_outlier[df_mask]
        # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def free_outlier(self, contamination=0.08, perplexity=20):
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        self.df_data = None
        df_outlier = self.df_get_outliers(contamination=contamination \
                                          , perplexity=perplexity)
        # df_free_outlier = self.df_data[df_outlier['OUTLIERS']==-1]
        index_inlier = [index for index in self.df_data.index if index not in df_outlier.index]
        df_free_outlier = self.df_data.loc[index_inlier]
        self.df_data = df_free_outlier

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def label_encode(self, df_label:pd.DataFrame):

        # Encode labels
        self._oLabelEncoder = LabelEncoder()

        # Align encoder with labels
        self._oLabelEncoder.fit(df_label[School.LABEL])

        # Transform labels into dataframe and store it in this instance
        self._df_label = pd.DataFrame(self._oLabelEncoder.transform(df_label[School.LABEL]), index=df_label.index)
        self._df_label.rename(columns={0: School.LABEL}, inplace=True)

        # Remove TARGET from dataset to vaoid data-leakage
        list_df_data = list(self.df_data.columns)
        if School.TARGET in list_df_data:
            list_df_data.remove(School.TARGET)
        else:
            pass
        self.df_data = pd.concat((self.df_data[list_df_data], self._df_label), axis=1)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def target_encode(self, method: str = 'binary') -> None:
        '''Encode target with labels ranged in [0,3] when not binary 
        else encode target with [0,1] labels.
        New column named School.LABEL is added into self.df_data dataframe/
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        def _three_labelization(value):
            if value <= 8:
                label = 'weak'
            elif 8 < value <= 12:
                label = 'middle'
            else:
                label = 'good'
            return pd.Series(label)

        def _four_labelization(value):
            if value <= 5:
                label = 'very_weak'
            elif 5 < value <= 10:
                label = 'weak'
            elif 10 < value <= 15:
                label = 'good'
            else:
                label = 'very_good'
            return pd.Series(label)

        def _binary_labelization(value):
            if value <= 11:
                label = 'weak'
            else:
                label = 'good'
            return pd.Series(label)

        if 'binary' == method:
            df_label = self.df_target[School.TARGET].apply(_binary_labelization)
        elif 'three' == method:
            df_label = self.df_target[School.TARGET].apply(_three_labelization)
        else:
            df_label = self.df_target[School.TARGET].apply(_four_labelization)

        df_label.rename(columns={0: School.LABEL}, inplace=True)

        self.label_encode(df_label)
        # Encode labels
        #self._oLabelEncoder = LabelEncoder()

        # Align encoder with labels
        #self._oLabelEncoder.fit(df_label[School.LABEL])

        # Transform labels into dataframe and store it in this instance
        #self._df_label = pd.DataFrame(self._oLabelEncoder.transform(df_label[School.LABEL]), index=df_label.index)
        #self._df_label.rename(columns={0: School.LABEL}, inplace=True)

        # Remove TARGET from dataset to vaoid data-leakage
        #list_df_data = list(self.df_data.columns)

        #list_df_data.remove(School.TARGET)
        #self.df_data = pd.concat((self.df_data[list_df_data], self._df_label), axis=1)

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def df_hist(self) -> None:
        '''Display histograms of features'''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        arrAxesSubplot = self.df_data.hist(xlabelsize=8, ylabelsize=8, figsize=(15, 15))
        [oAxesSubplot.title.set_size(10) for oAxesSubplot in arrAxesSubplot.ravel()]
        plt.show()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _df_data_scale(self, scaler, dict_scaler: Dict) -> pd.DataFrame:
        '''Scale quantitative features, transform qualitative features and 
        update df_data that is a dataframe concatenation of scaled quantitative 
        and dummy qualitative features.
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))
        # -------------------------------------------------------------------------------
        # Reset access to data
        # -------------------------------------------------------------------------------
        self.df_data = None

        # -------------------------------------------------------------------------------
        # Scale quantitative features
        # -------------------------------------------------------------------------------
        df_feature_quant = self.df_feature_quant
        self.oScaler = MyScaler(scaler(**dict_scaler) \
                                , list(df_feature_quant.columns) \
                                , self.logger)
        self.df_data = self.oScaler.fit_transform(self.df_data)

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def cluster_build(self \
                      , oClusterizer \
                      , scaler \
                      , dict_scaler: Dict = dict() \
                      , isDisplayed: bool = True
                      , perplexity: int = 25
                      , dict_display={'cmap': 'gray' \
                    , 'edgecolors': 'k' \
                    , 's': 60}) -> None:
        '''Build clusters from dataset.
        The variable School.TARGET is added to the list of quantitative features.
        Data are scaled.
        INPUT :
            * oClusterizer : instance of clusterization algorithm
            * scaler : class of transformer to scale data
            * dict_scaler : paramters for scaler
            * isDisplayed : when True, scatter of data points is plot in 2D, 
            points are colored with clusters labels
            * perplexity : for t-SNE metric

        OUTPUT : none

        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        # -------------------------------------------------------------------
        # Update list of quantitative features in order to take into account
        # FinalGrade feature while clustering
        # -------------------------------------------------------------------
        if School.TARGET not in self._list_feature_remove :
            self.list_feature_quantitative = self.list_feature_quantitative + [School.TARGET]

        # -------------------------------------------------------------------------------
        #
        # -------------------------------------------------------------------------------
        self._df_data_scale(scaler, dict_scaler)

        # -------------------------------------------------------------------------------
        # Apply clusterization over dataset
        # -------------------------------------------------------------------------------
        clusterization_name = oClusterizer.__str__()
        oClusterizer.fit(self.df_data.values)
        labels = oClusterizer.predict(self.df_data.values)

        # -------------------------------------------------------------------------------
        # Calculate Bayese and Akaike information criterias
        # -------------------------------------------------------------------------------
        aic = bic = -1.
        try:
            aic = np.round(oClusterizer.aic(self.df_data.values))
        except Exception as exception:
            self.logger.warning("cluster_build : AIC can not be calculated!")
        try:
            bic = np.round(oClusterizer.bic(self.df_data.values))
        except Exception as exception:
            self.logger.warning("cluster_build : BIC can not be calculated!")

        # -------------------------------------------------------------------------------
        # Update labels with cluster labels
        # -------------------------------------------------------------------------------
        df_label = pd.DataFrame(data=labels \
                                     , index=self.df_data.index \
                                     , columns=[School.LABEL])
        df_label.index.name = School.INDEX_STUDENT
        self.df_label = df_label
        # -------------------------------------------------------------------------------
        # Apply t-SNE with 2 dimensions reduction for display
        # -------------------------------------------------------------------------------
        if isDisplayed:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(self.df_data.values)
            self._df_data_proj = pd.DataFrame(data=X_tsne \
                                              , index=self.df_data.index \
                                              , columns=['Col1', 'Col2'])

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0] \
                                  , X_tsne[:, 1] \
                                  , c=self.df_label.values.ravel() \
                                  , **dict_display)

            legend = plt.legend(*scatter.legend_elements() \
                                , title="{} : clusters labels".format(clusterization_name) \
                                , loc="best")
            plt.gca().add_artist(legend)

            plt.title("t-SNE - Quantile transformation with perplexity= {}\
             (AIC, BIC)=({}, {})".format(perplexity, aic, bic))
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.show()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def gmm_cluster_build(self \
                          , nbgroup \
                          , scaler \
                          , dict_scaler \
                          , perplexity=20 \
                          , dict_display={'cmap': 'gray' \
                    , 'edgecolors': 'k' \
                    , 's': 60}):

        # -------------------------------------------------------------------
        # Reset dataset to the original values with no scaling
        # -------------------------------------------------------------------
        self.df_data = None

        # -------------------------------------------------------------------
        # Clusterization and labels update
        # -------------------------------------------------------------------
        oGaussianMixture = GaussianMixture(n_components=nbgroup \
                                           , random_state=13 \
                                           , verbose=1)
        self.cluster_build(oGaussianMixture \
                           , scaler
                           , dict_scaler=dict_scaler
                           , perplexity=perplexity \
                           , dict_display=dict_display)

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def get_scaled_train_test(self \
                              , test_size: float = 0.3 \
                              , scaler=IdentityScaler \
                              , dict_scaler: Dict = dict() \
                              , estimator: str = 'regressor' \
                              , withReduction: bool = False \
                              , dict_reduction: Dict = dict() \
                              ) -> List[pd.DataFrame]:
        '''
        1) Split dataset self.df_data as df_train, df_test
        2) scale quantitative features.
        3) Concatenate scaled quantitative features for both train and test dataset
        RETURN
            * df_X_train, df_y_train, df_X_test, df_y_test
        '''
        self.logger.info(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))

        self._estimator = estimator
        df_y = None

        if 'regressor' == self._estimator:
            list_col_target = list(self.df_target.columns)
            df_y = self.df_target
        elif 'classifier' == self._estimator:
            list_col_target = list(self.df_label.columns)
            df_y = self.df_label
        else:
            self.logger.warning(
                "get_scaled_train_test: unknown method= {} Fallback: regressor as default estimator".format(
                    self._estimator))
            self._estimator = 'regressor'
            list_col_target = self.df_target.columns
            df_y = self.df_target

        list_col_quant = list(self.df_feature_quant.columns)
        list_col_qual = list(self.df_dummy.columns)

        self.logger.debug("{} list_col_quant={}" \
                          .format(inspect.currentframe().f_code.co_name, list_col_quant))
        self.logger.debug("{} list_col_qual ={}" \
                          .format(inspect.currentframe().f_code.co_name, list_col_qual))

        list_col_reduction = list()
        if withReduction:
            self.dimension_reduction(dict_method=dict_reduction)
            list_col_reduction = [col for col in self.df_data.columns if col not in list_col_target]
            self.oScaler = None

        # --------------------------------------------------------------------
        # Split df_data
        # --------------------------------------------------------------------
        self.logger.debug("{} df_data.columns={}" \
                          .format(inspect.currentframe().f_code.co_name \
                                  , self.df_data.columns))

        df_train, df_test = train_test_split(self.df_data, \
                                             test_size=test_size, \
                                             random_state=13,\
                                             stratify=df_y)
        self.logger.debug("{} df_train.columns={}" \
                          .format(inspect.currentframe().f_code.co_name, df_train.columns))
        if withReduction:
            return df_train[list_col_reduction], df_train[list_col_target] \
                , df_test[list_col_reduction], df_test[list_col_target]
        else:
            # --------------------------------------------------------------------
            # Get qualitative features from dataset
            # --------------------------------------------------------------------
            df_train_qual = df_train[list_col_qual]
            df_train_qual = df_train_qual.transpose().drop_duplicates().transpose()

            df_test_qual = df_test[list_col_qual]
            df_test_qual = df_test_qual.transpose().drop_duplicates().transpose()

            # --------------------------------------------------------------------
            # Get quantitative features from dataset
            # --------------------------------------------------------------------
            df_train_quant = df_train[list_col_quant]
            df_train_quant = df_train_quant.transpose().drop_duplicates().transpose()

            df_test_quant = df_test[list_col_quant]
            df_test_quant = df_test_quant.transpose().drop_duplicates().transpose()

            self.logger.debug("{} list_col_quant={}" \
                              .format(inspect.currentframe().f_code.co_name \
                                      , list(df_train_quant.columns)))

            # --------------------------------------------------------------------
            # Concatenate quantitative and qualitative parts of 
            # train and test dataframes
            # --------------------------------------------------------------------
            df_train = pd.concat((df_train_quant, df_train_qual), axis=1)
            df_test = pd.concat((df_test_quant, df_test_qual), axis=1)

            if scaler is not None:
                # --------------------------------------------------------------------
                # Scale quantitative features
                # Fit the scaler with train data and transform train data
                # Scaling applies on quantitative features only
                # ----------------------------------------------------------------
                self.oScaler = MyScaler(scaler(**dict_scaler) \
                                        , list_col_quant \
                                        , self.logger)
                df_train_scaled = self.oScaler.fit_transform(df_train)

                # ---------------------------------------------------------------
                # Transform test data with scaler trained over train data
                # ----------------------------------------------------------------
                df_test_scaled = self.oScaler.transform(df_test)
            else:
                self.oScaler = None
                df_train_scaled = df_train
                df_test_scaled = df_test

            index_train = df_train_scaled.index
            index_test = df_test_scaled.index

            if 'regressor' == self._estimator:
                df_y_train = self.df_data.loc[index_train][list_col_target]
                df_y_test = self.df_data.loc[index_test][list_col_target]
            elif 'classifier' == self._estimator:
                df_y_train = self.df_label.loc[index_train][list_col_target]
                df_y_test = self.df_label.loc[index_test][list_col_target]
            else:
                self.logger.error("{} Unknown estimator={}" \
                                  .format(inspect.currentframe().f_code.co_name, self._estimator))
                df_y_train = df_y_test = pd.DataFrame()

            return df_train_scaled, df_y_train, df_test_scaled, df_y_test

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def get_best_classifier(self \
                            , scaler \
                            , dict_scaler \
                            , classifier \
                            , dict_classifier_grid \
                            , dict_calibration=None \
                            , perplexity=25 \
                            , dict_reduction=None
                            , is_displayed=True):
        '''Get best classifier considering classification metrics.

        Clusterization is performed based on number of clusters issued from clusters 
        exploration.
        Data transformation is applied with scaler.
        A grid search with cross-validation is performed in prder to yield the bests hyper-parameters.
        '''
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(School.LOGGER_PROMPT.format(function_name))
        self.oBestClassifierCalibrated = None
        self._is_calibrated = False

        # -------------------------------------------------------------------
        # Reset dataset to the original values with no scaling
        # -------------------------------------------------------------------
        self.df_data = None

        # -------------------------------------------------------------------
        # Split dataset over train and test 
        # Use projected value as new datasets whn withReduction flag 
        # is activated.
        # -------------------------------------------------------------------
        withReduction = False
        if dict_reduction is not None:
            withReduction = True
            scaler = None
        else:
            pass

        self.logger.debug("{} : scaler= {}".format(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name) \
                                                   , scaler))
        df_X_train, df_y_train, df_X_test, df_y_test \
            = self.get_scaled_train_test(scaler=scaler \
                                         , dict_scaler=dict_scaler
                                         , estimator='classifier' \
                                         , withReduction=withReduction \
                                         )

        # -------------------------------------------------------------------dict()
        # Create classifier
        # -------------------------------------------------------------------
        oClassifier = classifier()

        # -------------------------------------------------------------------
        # Search for best classifier and record it
        # -------------------------------------------------------------------
        dist_histogram = {'xlabelsize': 8, 'ylabelsize': 8, 'figsize': (15, 10), 'bins': 10}
        self.oBestClassifier = classifier_grid_crossval(oClassifier \
                                                        , self.oScaler \
                                                        , df_X_train \
                                                        , df_y_train \
                                                        , df_X_test \
                                                        , df_y_test \
                                                        , dict_param_grid=dict_classifier_grid \
                                                        , is_displayed=False)
        # , dict_hist_display=dist_histogram)

        self.dict_df_X_train_test = {"df_X_train": df_X_train \
            , "df_y_train": df_y_train \
            , "df_X_test": df_X_test \
            , "df_y_test": df_y_test}

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def classifier_calibrate(self, dict_grid_calibration):
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(School.LOGGER_PROMPT.format(function_name))

        if self.oBestClassifier is None:
            self.logger.error(School.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name) + ": None Classifier!")
        else:
            df_X_test = self.dict_df_X_train_test['df_X_test']
            df_y_test = self.dict_df_X_train_test['df_y_test']

            oCalibratedClassifierCV = CalibratedClassifierCV(self.oBestClassifier)
            oRepeatedStratifiedKFold = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=13)
            oGridSearchCV = GridSearchCV(oCalibratedClassifierCV \
                                         , dict_grid_calibration \
                                         , scoring='f1_macro' \
                                         , cv=oRepeatedStratifiedKFold \
                                         , error_score='raise'
                                         , verbose=0)
            try :
                grid_result = oGridSearchCV.fit(df_X_test.values, df_y_test.values.ravel())

                print("Best score: {:.2f} Parameters: {}" \
                    .format(grid_result.best_score_, grid_result.best_params_))

                self.oBestClassifierCalibrated = grid_result.best_estimator_
                self._is_calibrated = True
            except Exception as exception:
                self.logger.error(Common.LOGGER_PROMPT+" calibration failed: {}".format(function_name, exception))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def feature_remove(self, feature):
        method_name = inspect.currentframe().f_code.co_name
        self.logger.info(School.LOGGER_PROMPT + ": {}".format(method_name, feature))

        if feature in self._list_feature_qualitative:
            list_feature_qualitative = self._list_feature_qualitative
            list_feature_qualitative.remove(feature)
            self.list_feature_qualitative = list_feature_qualitative
        else:
            pass

        if feature in self._list_feature_quantitative:
            list_feature_quantitative = self._list_feature_quantitative
            list_feature_quantitative.remove(feature)
            self.list_feature_quantitative = list_feature_quantitative
        else:
            pass
        self._list_feature_remove.append(feature)

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Propreties
    # --------------------------------------------------------------------------
    @property
    def df_student(self):
        return self._df_student

    @df_student.setter
    def df_student(self, df):
        message = " Forbidden operation: update"
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def analysis_file(self):
        return self._html_profile

    @analysis_file.setter
    def analysis_file(self, html_profile):
        self._html_profile = html_profile

    @property
    def df_identity(self):
        return self._df_identity

    @df_identity.setter
    def df_identity(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_feature(self):
        return self._df_feature

    @df_feature.setter
    def df_feature(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def list_feature_quantitative(self):
        return list(self._list_feature_quantitative)

    @list_feature_quantitative.setter
    def list_feature_quantitative(self, list_feature):
        self.df_data = None
        # --------------------------------------------------------------------------------
        # The use of set avoid accidental features duplication
        # --------------------------------------------------------------------------------
        list_feature = list(set(list_feature))
        self._list_feature_quantitative = list_feature.copy()
        self.logger.info(School.LOGGER_PROMPT + "{}: updated".format(inspect.currentframe().f_code.co_name))

    @property
    def list_feature_qualitative(self):
        return list(self._list_feature_qualitative)

    @list_feature_qualitative.setter
    def list_feature_qualitative(self, list_feature):
        self.df_data = None
        # --------------------------------------------------------------------------------
        # The use of set avoid accidental features duplication
        # --------------------------------------------------------------------------------
        list_feature = list(set(list_feature))
        self._list_feature_quantitative = list_feature.copy()
        self.logger.info(School.LOGGER_PROMPT \
                         + "{}: updated".format(inspect.currentframe().f_code.co_name))

    @property
    def list_col_feature(self):
        return list(self._df_dummy.columns) + list(self._list_feature_quantitative)

    @list_col_feature.setter
    def list_col_feature(self, list_col):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_feature_quant(self):
        return self._df_feature[self.list_feature_quantitative]

    @df_feature_quant.setter
    def df_feature_quant(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_feature_qual(self):
        return self._df_feature[self._list_feature_qualitative]

    @df_feature_qual.setter
    def df_feature_qual(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_target(self):
        return self._df_feature[[School.TARGET]]

    @df_target.setter
    def df_target(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_dummy(self):
        return self._df_get_dummy_feature()

    @df_dummy.setter
    def df_dummy(self, df):
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def df_data(self):
        '''Concatenate quantitative and qualitative features and return 
        the dataframe.
        '''
        if self._df_data is None:
            df_feature = pd.concat((self.df_feature_quant, self.df_dummy), axis=1)
            self._df_data = pd.concat((df_feature, self.df_target), axis=1)
            return self._df_data
        else:
            return self._df_data

    @df_data.setter
    def df_data(self, df):
        '''Allows to reset df_data that contains all 
        features except target.
        df_data is used to split dataset as train and test datasets.
        '''
        if df is None:
            self._df_data = None
            self.logger.info("df_data: updated with None")
        else:
            self._df_data = df.copy()
            self.logger.info("df_data: updated with dataframe shape={}".format(self._df_data.shape))

    @property
    def df_label(self) -> pd.DataFrame:
        return self._df_label

    @df_label.setter
    def df_label(self, df: pd.DataFrame) -> None:
        '''Save given dataframe into self._df_label attribute
        Replace self._df_label with the given dataframe.
        Concatenate the given dataframe with self._df_data
        NB: Given dataframe is supposed to have same index then self._df_data
        
        '''
        self._df_label = df.copy()
        colname = df.columns[-1]
        self.df_data = pd.concat((self.df_data, self.df_label), axis=1)
        self.logger.info("df_label: updated with a dataframe column name={}".format(colname))

    @property
    def df_data_proj(self) -> pd.DataFrame:
        return self._df_data_proj

    @df_data_proj.setter
    def df_data_proj(self, df: pd.DataFrame) -> None:
        self.logger.error(School.LOGGER_PROMPT \
                          + School.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))

    @property
    def oScaler(self):
        return self._oScaler

    @oScaler.setter
    def oScaler(self, oScaler) -> None:
        self._oScaler = oScaler
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))

    @property
    def oBestClassifier(self):
        return self.oBestClassifierCalibrated

    @oBestClassifier.setter
    def oBestClassifier(self, oBestClassifier) -> None:
        self._oBestClassifier = oBestClassifier
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))

    @property
    def oBestClassifierCalibrated(self):
        if self._oBestClassifierCalibrated is None:
            return self._oBestClassifier
        else:
            pass
        return self._oBestClassifierCalibrated

    @oBestClassifierCalibrated.setter
    def oBestClassifierCalibrated(self, classifier) -> None:
        self._oBestClassifierCalibrated = classifier
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))

    @property
    def dict_df_X_train_test(self):
        return self._dict_df_X_train_test

    @dict_df_X_train_test.setter
    def dict_df_X_train_test(self, dict_df_X_train_test: Dict) -> None:
        self._dict_df_X_train_test = dict_df_X_train_test.copy()
        self.logger.info("{}".format(inspect.currentframe().f_code.co_name))
    @property
    def is_calibrated(self):
        return self._is_calibrated

    @is_calibrated.setter
    def is_calibrated(self, is_calibrated: bool) -> None:
        pass

    # -------------------------------------------------------------------------------
