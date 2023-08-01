# -*- coding: utf-8 -*-

'''This module contains utilities to explain models used on School instance'''
import sys
import inspect
import os
import plotly.express as px

import shap
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

import random
from core.school import School
from typing import Dict

from util.common import Common
from core.scaler import MyScaler, IdentityScaler

class SchoolExplainer(Common):
    DICT_FEATUREIMPROVE = {'school_MS': 0,
                           'sex_M': 0,
                           'address_U': 1,
                           'famsize_LE3': 0,
                           'Pstatus_T': 0,
                           'Mjob_health': 0,
                           'Mjob_other': 0,
                           'Mjob_services': 0,
                           'Mjob_teacher': 0,
                           'Fjob_health': 0,
                           'Fjob_other': 0,
                           'Fjob_services': 0,
                           'Fjob_teacher': 0,
                           'reason_home': 1,
                           'reason_other': 1,
                           'reason_reputation': 1,
                           'guardian_mother': 1,
                           'guardian_other': 1,
                           'schoolsup_yes': 1,
                           'famsup_yes': 1,
                           'paid_yes': 1,
                           'activities_yes': 1,
                           'nursery_yes': 0,
                           'higher_yes': 1,
                           'internet_yes': 1,
                           'romantic_yes': 1,
                           'age': 0,
                           'Medu': 0,
                           'Fedu': 0,
                           'traveltime': 1,
                           'studytime': 1,
                           'failures': 0,
                           'famrel': 1,
                           'freetime': 1,
                           'goout': 1,
                           'Dalc': 1,
                           'Walc': 1,
                           'health': 1,
                           'absences': 1,
                           'FinalGrade': 0}
    SCORE_NAME = "ImprovabilityScore"

    def __init__(self, oSchool: School) -> None:
        super().__init__()
        self._df_y = None
        self._df_X = None
        self._oSchool = oSchool
        self._shap_values = None
        self._dataframe_selection = ""
        self._df_X_improve = None
        self._df_shap_values_improve = None
        self._class_label = None
        self.explainer = None
        self.list_improve_pos = list()

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def get_shap_explainer(self, classifier, df_X:pd.DataFrame) -> shap.Explainer:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))

        explainer = None
        if classifier is not None:
            if isinstance(classifier, XGBClassifier) or \
                    isinstance(classifier, RandomForestClassifier) or\
                    isinstance(classifier, DecisionTreeClassifier):
                try:
                    explainer = shap.TreeExplainer(classifier.predict_proba, df_X)
                    self.logger.debug("{} explainer= shap.TreeExplainer".format(function_name))
                except Exception as exception:
                    message = "Tried TreeExplainer produced: {}.\n Fallback to shap.Explainer!".format(exception)
                    self.logger.warning(Common.LOGGER_PROMPT+" {}".format(function_name, message))
                    explainer = shap.Explainer(classifier.predict_proba, df_X)                    
            elif isinstance(classifier, LogisticRegression):
                self.logger.debug("{} explainer= shap.LinearExplainer".format(function_name))
                explainer = shap.Explainer(classifier.predict_proba, df_X)
            elif isinstance(classifier, CalibratedClassifierCV):
                self.logger.debug("{} explainer= shap.Explainer".format(function_name))
                explainer = shap.Explainer(classifier.predict_proba, df_X)
            else:
                try:
                    explainer = shap.KernelExplainer(classifier.predict_proba, df_X)
                    self.logger.debug("{} explainer= shap.Explainer".format(function_name))
                except Exception as exception:
                    message = "Tried KernelExplainer produced: {}.\n Fallback to shap.Explainer!".format(exception)
                    self.logger.warning(Common.LOGGER_PROMPT+" {}".format(function_name, message))
                    explainer = shap.Explainer(classifier.predict_proba, df_X)
                    self.logger.debug("{} explainer= shap.Explainer".format(function_name))
        else:
            self.logger.error("{} : classifier is None!".format(function_name))

        self.logger.debug("{} explainer={} df_X.shape={}".format(function_name, explainer, df_X.shape))
        return explainer
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def calculate_shapvalues(self, dataframe_selection='train_test'):
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))

        classifier = self._oSchool.oBestClassifierCalibrated
        if classifier is None:
            self.logger.error("{} : classifier is None".format(function_name))
        else:
            df_X = None
            df_y = None
            df_X_train = None
            df_X_test = None
            explainer = None
            self._dataframe_selection = dataframe_selection

            df_X_train = self._oSchool.dict_df_X_train_test['df_X_train']
            df_X_test = self._oSchool.dict_df_X_train_test['df_X_test']
            df_y_train = self._oSchool.dict_df_X_train_test['df_y_train']
            df_y_test = self._oSchool.dict_df_X_train_test['df_y_test']
            if 'train_test' == self.dataframe_selection:
                pass
            elif 'train' == self.dataframe_selection:
                df_X_test = df_X_train
                df_y_test = df_y_train
            elif 'test' == self.dataframe_selection:
                df_X_train = df_X_test
                df_y_train = df_y_test
            elif 'all' == self.dataframe_selection:
                df_X = pd.concat((df_X_train, df_X_test), axis=0)
                df_y = pd.concat((df_y_train, df_y_test), axis=0)
            else:
                message = "supported dataframe_selection modes are: train_test, train, test, all;\
                 fallback to train_test mode"
                self.logger.warning("{} : {}".format(function_name, message))
                self._dataframe_selection = "train_test"
            if df_X is None:
                df_X = df_X_train
                df_y = df_y_train
            else:
                df_X_test  = df_X
                df_y_test = df_y

            explainer = self.get_shap_explainer(classifier, df_X)

            # ----------------------------------------------------------------------
            # Shunt standard output and standard error for avoiding warnings
            # ----------------------------------------------------------------------
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            self._shap_values = explainer(df_X_test)
            sys.stderr = old_stderr

            self.explainer = explainer
            self._df_X = df_X
            self._df_y = df_y
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def calculate_improvability(self, class_label, oScaler=IdentityScaler()):
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))

        df_X = self.df_X
        self._class_label = class_label
        # ----------------------------------------------------------------------
        # For each classe, calculate the improvability score for any student.
        # ----------------------------------------------------------------------
        list_feature_improvable = [feature_improve \
                                   for feature_improve, is_improvable in SchoolExplainer.DICT_FEATUREIMPROVE.items() \
                                   if 1 == is_improvable]
        self.list_improve_pos = sorted(df_X.columns.get_loc(feature) for feature in list_feature_improvable)
        df_shap_values = pd.DataFrame(data=self.shap_values.values[:, :, self._class_label] \
                                        , columns=df_X.columns,
                                        index=df_X.index)
        
        df_improve = df_shap_values[list_feature_improvable].copy()
        df_improve_score = df_improve.sum(axis=1).to_frame()
        improve_score_scaled = oScaler.fit_transform(df_improve_score.values)
        df_improve_score = pd.DataFrame(data=improve_score_scaled\
            , columns=df_improve_score.columns \
            , index=df_improve_score.index)

        col_to_rename = df_improve_score.columns[0]
        df_improve_score.rename(columns={col_to_rename: SchoolExplainer.SCORE_NAME}, inplace=True)

        self._df_X_improve = pd.concat((df_X, df_improve_score), axis=1)
        self._df_shap_values_improve = pd.concat((df_shap_values, df_improve_score), axis=1)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def check_consistency(self) -> bool:
        """Check that indexes from shapley values array
        and dataframe used to build shapley values do reference
        same records.

        For doing so, a column and index are randomly selected from df_XoSchoolExplainer.shap_values
        A dataframe is constructed from shapley array with same indexes and
        columns than df_X
        Values issued from randomly selected index and column are compared for
        dataframe shapley values and dataframe used to construct shapley values.
        """
        is_consistent = False
        function_name = inspect.currentframe().f_code.co_name
        # self.logger.info(SchoolExplainer.LOGGER_PROMPT.format(function_name))
        if 'train_test' == self.dataframe_selection:
            df_X = self._oSchool.dict_df_X_train_test['df_X_test']
        else:
            df_X = self.df_X

        random_label = random.choice(self._oSchool.df_label.loc[df_X.index][School.LABEL].unique())
        df_shap_data = pd.DataFrame(data=self.shap_values[:, :, random_label].data \
                                    , index=df_X.index \
                                    , columns=df_X.columns)
        random_feature = random.choice(df_X.columns)
        random_index = random.choice(df_X.index)
        pos_col = df_X.columns.get_loc(random_feature)
        is_consistent = df_X.loc[random_index][random_feature] == df_shap_data.loc[random_index][random_feature]
        if not is_consistent:
            # return df_X.loc[random_index][pos_col], df_shap_.loc[random_index][pos_col]
            # self.logger.info("{} : consistency is {}".format(function_name, is_consistent))
            pass
        else:
            # self.logger.info("{} : consistency is {}".format(function_name, is_consistent))
            pass
        return is_consistent

    # --------------------------------------------------------------------------

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------------
    def dependence_plot_classes_new(self \
                                    , x_feature_name \
                                    , y_feature_name \
                                    , class_label \
                                    , is_improve=False \
                                    , legend="Shapley values for" \
                                    , width=500 \
                                    , height=500 \
                                    , facet_col=None) -> pd.DataFrame:
        """
        Create a plot showing relationship between a feature displayed along
        horizontal axis and the shapley value of another feature displayed along
        Y axis.

        In addition, distributions of features values related to shapley values
        are displayed on graph margins.

            INPUT
            * x_feature_name : str or int
                Indicate name of the feature from which values are
                displayed along X-axis

            * y_feature_name : str or int
                Indicate name of the feature from which shapley values are
                displayed along y-axis. The values of this feature name are also
                displayed in the graph legend

            * class_label : class value used to select Shapley values from shap.Explanation.

            * list_id : list
                the list of identifiers

            * facet_col : name of the feature that will react as a facet.
        """
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))
        df_X = None
        df_shap_values = None

        # ---------------------------------------------------------------------------
        # Get dataframe matching with Shapley values calculation.
        # Shapley values calculation depends on method used to select dataframe.
        # ---------------------------------------------------------------------------
        if not is_improve:
            if 'train_test' == self.dataframe_selection:
                df_X = self._oSchool.dict_df_X_train_test['df_X_test']
            else:
                df_X = self.df_X
            # ---------------------------------------------------------------------------
            # Create dataframe based on Shapley values that contains all features to
            # be displayed along (x,y) axis.
            # For binary classification, class_label is either 0 or 1.
            # ---------------------------------------------------------------------------
            df_shap_values = pd.DataFrame(data=self.shap_values.values[:, :, class_label] \
                                          , columns=df_X.columns, \
                                          index=df_X.index)
            max_display = df_shap_values.shape[0]
        else:
            df_X = self._df_X_improve
            df_shap_values = self._df_shap_values_improve
            max_display = len(self.list_improve_pos)

        if x_feature_name not in df_shap_values.columns:
            if x_feature_name in self._oSchool.df_target.columns:
                df_shap_values = pd.concat((df_shap_values \
                                                , self._oSchool.df_target[[x_feature_name]].loc[df_X.index]), axis=1)
            elif x_feature_name in self._oSchool.df_label.columns:
                df_shap_values = pd.concat((df_shap_values \
                                                , self._oSchool.df_label[[x_feature_name]].loc[df_X.indx]), axis=1)
            else:
                message = "Can't find {} if df_target nor df_label dataframes"
                # self.logger.error(Common.LOGGER_PROMPT+": {}".format(function_name, message))
                return pd.DataFrame()

        x_values = df_shap_values[x_feature_name].loc[df_shap_values.index].values
        y_feature_name_legend = y_feature_name

        # ---------------------------------------------------------------------------
        # Get values for y_feature_name; they are used to color points
        # ---------------------------------------------------------------------------
        y_values = df_X[y_feature_name].values

        # ---------------------------------------------------------------------------
        # Get shap values for y_feature_name; They are used to position points on
        # y axis.
        # ---------------------------------------------------------------------------
        y_shap_values = df_shap_values[y_feature_name].values

        # ---------------------------------------------------------------------------
        # Identifier are displayed
        # ---------------------------------------------------------------------------
        list_id = df_shap_values.index

        # ---------------------------------------------------------------------------
        # Build dataframe that contains content to be displayed
        # ---------------------------------------------------------------------------
        y_shap_values_name = legend + ' ' + y_feature_name
        dict_df = { \
            x_feature_name: x_values \
            , y_feature_name_legend: y_values \
            , y_shap_values_name: y_shap_values \
            , 'StudentID': list_id \
            }
        df_display = pd.DataFrame(dict_df)

        if facet_col is not None:
            df_display = pd.concat((df_display, df_X[facet_col].reset_index()), axis=1)
            fig = px.scatter(df_display \
                             , x=x_feature_name \
                             , y=y_shap_values_name \
                             , color=y_feature_name_legend \
                             , hover_data=['StudentID'] \
                             # , marginal_y="histogram"\
                             # , marginal_x="histogram"\
                             , facet_col='Clusters' \
                             , width=width \
                             , height=height \
                             , opacity=0.5)
        else:
            fig = px.scatter(df_display \
                             , x=x_feature_name \
                             , y=y_shap_values_name \
                             , color=y_feature_name_legend \
                             , hover_data=['StudentID'] \
                             , marginal_y="histogram" \
                             , marginal_x="histogram" \
                             , width=width \
                             , height=height \
                             , trendline="ols" \
                             , opacity=0.5)

        fig.update_xaxes(autorange="reversed")
        fig.show(figsize=(12, 12))
        return df_display

    # -------------------------------------------------------------------------------
    @property
    def shap_values(self) -> shap.Explanation:
        return self._shap_values

    @shap_values.setter
    def shap_values(self, arr: shap.Explanation) -> None:
        # self.logger.error(Common.LOGGER_PROMPT \
        #                  + Common.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))
        pass

    @property
    def df_X(self) -> pd.DataFrame:
        """Returns dataframe depending on method """
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))
        df_X = None
        if 'train_test' == self.dataframe_selection:
            df_X = self._oSchool.dict_df_X_train_test['df_X_test']
        else:
            df_X = self._df_X
        return df_X

    @df_X.setter
    def df_X(self, df: pd.DataFrame) -> None:
        pass

    @property
    def df_y(self) -> pd.DataFrame:
        """Returns dataframe depending on method """
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info("{}".format(function_name))
        return self._df_y

    @df_y.setter
    def df_y(self, df: pd.DataFrame) -> None:
        # self.logger.error(Common.LOGGER_PROMPT \
        #                  + Common.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))
        pass

    @property
    def dataframe_selection(self) -> str:
        return self._dataframe_selection

    @dataframe_selection.setter
    def dataframe_selection(self, string) -> None:
        # self.logger.error(Common.LOGGER_PROMPT \
        #                  + Common.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))
        pass

    @property
    def dict_label_index(self) -> Dict:
        return self._oSchool.get_dict_label_index()

    @dict_label_index.setter
    def dict_label_index(self, dict_) -> None:
        # self.logger.error(Common.LOGGER_PROMPT \
        #                  + Common.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))
        pass

    @property
    def class_label(self) -> Dict:
        function_name = inspect.currentframe().f_code.co_name
        message = "Label= {}".format(self._class_label)
        self.logger.debug(Common.LOGGER_PROMPT + " {}".format(function_name, message))
        return self._class_label

    @dict_label_index.setter
    def dict_label_index(self, class_label) -> None:
        # self.logger.error(Common.LOGGER_PROMPT \
        #                  + Common.SETTER_MESSAGE_ERROR.format(inspect.currentframe().f_code.co_name))
        pass
