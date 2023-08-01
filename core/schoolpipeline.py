"""This module contains components to process the data workflow"""

import pandas as pd
import matplotlib.pyplot as plt

import inspect
import shap
import logging


from util.analysis import pr_display, roc_display
from util.common import Common, halt
from core import config as core_config
from core.dataloader import DataLoader
from core.school import School
from core.schoolexplainer import SchoolExplainer

import core.config as config_core

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
class SchoolPipeline(Common):
    INPUT_MESSAGE = "Entrez l'identifiant de l'étudiant (Return pour finir): "
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._label_mode=None
        self._oDataLoader = None
        self._oSchool = None       
        self._oSchoolExplainer = None
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def load_data(self, source=None) -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))
        
        self._oDataLoader = DataLoader(dataSource=core_config.DATA_SOURCE)
        #df_student = oDataLoader.df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def labelization(self, label_mode="encode") -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))

        self._label_mode = label_mode
        self._oSchool = School(self._oDataLoader.df)
        self._oSchool.df_data = None
        if "encode" == self._label_mode:
            self._oSchool.target_encode()
        else:
            message = "Labellisation mode not supported: {}".format(self._label_mode)
            self.logger.error(Common.LOGGER_PROMPT.format(message))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def classify(self\
        , classifier\
        , dict_classifier_grid\
        , scaler=config_core.MyScaler) -> None:

        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))
        self.logger.debug("{}: classifier={}".format(function_name, classifier.__name__))

        self._oSchool.get_best_classifier(  scaler\
                                    , dict()\
                                    , classifier\
                                    , dict_classifier_grid\
                                    , dict_calibration=None\
                                    , perplexity=25\
                                    , dict_reduction=None\
                                    , is_displayed=True)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def evaluate(self) -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))

        # Quantitative features are scaled with scaler trained on train dataset
        df_X_test = self._oSchool.dict_df_X_train_test['df_X_test']
        df_y_test = self._oSchool.dict_df_X_train_test['df_y_test']
        
        if self._oSchool.df_label is not None:
            if 2 == len(self._oSchool.df_label[School.LABEL].unique()):                
                roc_display(self._oSchool.oBestClassifier, df_X_test.values, df_y_test.values.ravel())
                pr_display(self._oSchool.oBestClassifier, self._oSchool.df_label, df_X_test.values, df_y_test.values.ravel())
            else:
                pass
            dict_label_index = self._oSchool.get_dict_label_index()
            df_student = self._oDataLoader.df
            for label, index in dict_label_index.items():
                count = len(index)
                self.logger.info("Group: {} Count={} FinalGrade Average={:.2f}".format(label\
                                            , count
                                            , df_student.loc[index][School.TARGET].mean()))
        else:
            self.logger.error(Common.LOGGER_PROMPT.format("No dataframe issued from labellisation!"))
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def calibrate(self) -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))
        
        dict_grid_calibration = dict(cv=[2,3,4], method=['sigmoid','isotonic'])
        self._oSchool.classifier_calibrate(dict_grid_calibration)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def chap_values(self, dataframe_selection='train_test') -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))

        oSchoolExplainer = SchoolExplainer(self._oSchool)
        oSchoolExplainer.calculate_shapvalues(dataframe_selection=dataframe_selection)
        self.logger.info("{}".format(oSchoolExplainer.shap_values.shape))
        self.logger.info("{}".format(oSchoolExplainer.check_consistency()))  
        self._oSchoolExplainer = oSchoolExplainer

    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def explain(self, title="") -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))
        
        list_improve_pos = self._oSchoolExplainer.list_improve_pos
        df_X = self._oSchoolExplainer.df_X
        
        if 0 < len(list_improve_pos):
            shap_values\
                = self._oSchoolExplainer.shap_values[:, list_improve_pos, :]
            list_feature = [feature for feature in [df_X.columns[pos] for pos in list_improve_pos]]
        else:
            shap_values = self._oSchoolExplainer.shap_values
            list_feature = list(df_X.columns)

        class_label = self._oSchoolExplainer._class_label
        if class_label is None:
            class_label = 1
        else:
            pass
        shap.summary_plot(shap_values[:, :, class_label]\
            , df_X[list_feature])
        if 0 < len(title):
            plt.title(title)
            plt.show()
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def calculate_improvability(self, oScaler, class_label=1) -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))

        self._oSchoolExplainer.calculate_improvability(class_label, oScaler)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def dependence_plot(self, x_feature_name, y_feature_name) -> None:
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))
        
        class_label = self._oSchoolExplainer.class_label
        _ = self._oSchoolExplainer.dependence_plot_classes_new(\
                            x_feature_name\
                            , y_feature_name\
                            , class_label\
                            , is_improve=True
                            , legend="Shapley sum "
                            , width=800
                            , height=800
                            , facet_col=None)   
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def process_binary_classification(self):
        function_name = inspect.currentframe().f_code.co_name
        self.logger.info(Common.LOGGER_PROMPT.format(function_name))

        dict_classifier_grid = core_config.dict_classifier_grid
        classifier = config_core.CLASSIFIER
        scaler = config_core.SCALER
        scaler_improve = config_core.SCALER_IMPROVE
        dataframe_selection = config_core.DATAFRAME_SELECTION
        class_label = config_core.CLASS_LABEL

        self.load_data()
        self.labelization()
        
        try:
            classifier_name = classifier.__name__
            if classifier_name not in dict_classifier_grid.keys():
                halt("Unknown classifier in registry: {}".format(classifier_name))
            self.classify(classifier, dict_classifier_grid[classifier_name]\
                , scaler=scaler)
        except Exception as exception:
            self.logger.critic(classifier().get_params().keys())
            halt(self.logger, exception)

        self.evaluate()
        if config_core.HAS_CALIBRATION:
            self.calibrate()
            self.evaluate()
        
        self.chap_values(dataframe_selection=dataframe_selection)
        self.explain()

        self.calculate_improvability(scaler_improve(), class_label=class_label)
        x_feature_name = School.TARGET
        y_feature_name = SchoolExplainer.SCORE_NAME
        self.dependence_plot(x_feature_name, y_feature_name)
    #--------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def _shap_values_data_inverse_transform(self):
        list_feature_quantitative = self._oSchool.list_feature_quantitative
        df_X = self._oSchoolExplainer.df_X
        df_X_quant = df_X[list_feature_quantitative]

        shap_values = self.oSchoolExplainer.shap_values

        df_shapvalues = pd.DataFrame(data=shap_values.data[:, :df_X_quant.shape[1]]
                                     , columns=df_X_quant.columns[:df_X.shape[1]]
                                     , index=df_X_quant.index)
        X_quant = self._oSchool.oScaler.scaler.inverse_transform(df_shapvalues)

        df_quant = pd.DataFrame(data=X_quant
                                , columns=df_X_quant.columns[:df_X_quant.shape[1]]
                                , index=df_X_quant.index)

        df_qual = pd.DataFrame(data=shap_values.data[:, df_X_quant.shape[1]:]
                               , columns=df_X.columns[df_X_quant.shape[1]:]
                               , index=df_X_quant.index)
        df = pd.concat((df_quant, df_qual), axis=1)
        shap_values.data = df.values
        return shap_values, df

    # --------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #--------------------------------------------------------------------------
    def interactive(self):
        isTrue = True
        shap_values, _ = self._shap_values_data_inverse_transform()
        #shap_values = self.oSchoolExplainer.shap_values
        df_X = self._oSchoolExplainer.df_X
        class_label = self.oSchoolExplainer.class_label
        list_improve_pos = self.oSchoolExplainer.list_improve_pos
        #list_improve_pos = list(self.oSchoolExplainer.df_X.columns)

        shap_values = shap_values[:, list_improve_pos, :]
        while isTrue:
            studentID = input(SchoolPipeline.INPUT_MESSAGE)
            if 0 == len(studentID):
                isTrue = False
            else:
                try:
                    studentID = int(studentID)
                    try:
                        firstName = self._oDataLoader.df['FirstName'].loc[studentID]
                        lastName = self._oDataLoader.df['FamilyName'].loc[studentID]
                        finalGrade = self._oDataLoader.df[School.TARGET].loc[studentID]
                        mask = df_X.reset_index()['index'] == studentID
                        shap_index = df_X.reset_index()[mask].index[0]

                        shap.plots.waterfall(shap_values[shap_index][:, class_label],\
                                             max_display=len(list_improve_pos))
                        message = "{0:} {1:} "+School.TARGET+"={2:}"
                        print(message.format(firstName, lastName, finalGrade))
                    except Exception as exception:
                        logging.error(exception)
                except Exception as exception:
                    logging.error(exception)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Properties
    #--------------------------------------------------------------------------
    @property
    def oSchoolExplainer(self) -> SchoolExplainer:
        return self._oSchoolExplainer

    @oSchoolExplainer.setter
    def oSchoolExplainer(self, explainer:SchoolExplainer) -> None:
        self.logger.error(Common.LOGGER_PROMPT.format(inspect.currentframe().f_code.co_name))
        self._oSchoolExplainer = explainer
    #--------------------------------------------------------------------------
