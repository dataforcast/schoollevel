# -*- coding: utf-8 -*-

'''This module allows to configure the application behavior'''

import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import seaborn as sns
import plotly.io as pio

pio.templates.default = "presentation"


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def corr_matrix_display(corrmatrix: pd.DataFrame \
                        , method='Pearson' \
                        , figsize: Tuple = (15, 15)) -> None:
    # ---------------------------------------------------------------------------
    # Build mask for lower symetric part
    # ---------------------------------------------------------------------------
    mask = np.zeros_like(corrmatrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # ---------------------------------------------------------------------------
    # Build matrix color map
    # ---------------------------------------------------------------------------
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # ---------------------------------------------------------------------------
    # Display result
    # ---------------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.heatmap(corrmatrix, \
                mask=mask, \
                cmap=cmap, \
                annot=True, \
                fmt='.2f', \
                square=True, \
                linewidths=.5, \
                cbar_kws={"shrink": .5})
    plt.title('Correlation matrix - {}'.format(method))
    plt.show()


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def regressor_grid_crossval(oRegressor \
                            , df_X_train \
                            , df_y_train \
                            , df_X_test \
                            , df_y_test \
                            , dict_param_grid=dict() \
                            , is_displayed=True
                            , dict_hist_display=dict()):
    '''Scores a regressor estimator. 
    '''
    oGridSearchCV = GridSearchCV(oRegressor \
                                 , dict_param_grid \
                                 , scoring='neg_mean_squared_error' \
                                 , cv=3, verbose=1)
    oGridSearchCV.fit(df_X_train.values, df_y_train.values.ravel())

    # Bests parameters
    print("Bests parameters :")
    print(oGridSearchCV.best_params_)

    # Instantiate the best model 
    oBestRgressor = oGridSearchCV.best_estimator_
    y_pred = oBestRgressor.predict(df_X_test.values)

    #  Scoring the best model

    y_test = df_y_test.values.ravel()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE over test dataset :", mse)
    print("MAE over test dataset :", mae)
    print("R² over test dataset  :", r2)
    if is_displayed:
        df_test_pred = pd.DataFrame(data={'true': y_test \
            , 'pred': y_pred}, index=df_y_test.index)
        # histograms of the variables
        arrAxesSubplot = df_test_pred.hist(**dict_hist_display)
        [oAxesSubplot.title.set_size(10) for oAxesSubplot in arrAxesSubplot.ravel()]
        plt.show()
    return oBestRgressor


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def display_classification_hist(classifier \
                                , df_X_test \
                                , df_y_test \
                                , dict_hist_display):
    y_pred = classifier.predict(df_X_test.values)
    df_test_pred = pd.DataFrame(data={'true': df_y_test.values.ravel() \
        , 'pred': y_pred.ravel()} \
                                , index=df_y_test.index)
    # histograms of the variables
    arrAxesSubplot = df_test_pred.hist(**dict_hist_display)
    [oAxesSubplot.title.set_size(10) for oAxesSubplot in arrAxesSubplot.ravel()]
    plt.show()


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def classifier_grid_crossval(oClassifier \
                             , scaler \
                             , df_X_train \
                             , df_y_train \
                             , df_X_test \
                             , df_y_test \
                             , dict_param_grid=dict() \
                             , is_displayed=True
                             , dict_hist_display=dict()):
    '''Scores a classifier. 
    '''
    f1_scorer = make_scorer(f1_score, average='weighted')
    precision_scorer = make_scorer(precision_score, average='weighted')
    recall_scorer = make_scorer(recall_score, average='weighted')

    dict_scoring = {
        'f1': f1_scorer,
        'precision': precision_scorer,
        'recall': recall_scorer
    }
    oStratifiedKFold = StratifiedKFold(n_splits=3 \
                                       , shuffle=True \
                                       , random_state=13)

    oGridSearchCV = GridSearchCV(oClassifier \
                                 , dict_param_grid \
                                 , scoring=f1_scorer \
                                 , refit='f1' \
                                 , cv=oStratifiedKFold \
                                 , verbose=0)
    oGridSearchCV.fit(df_X_train.values, df_y_train.values.ravel())

    # Bests parameters
    print("Bests parameters :")
    print(oGridSearchCV.best_params_)

    # Instantiate the best model 
    oBestClassifier = oGridSearchCV.best_estimator_

    # Quantitative features are scaled with scaler trained on train dataset
    if scaler is not None:
        df_X_test = scaler.transform(df_X_test)

    list_metric = ['precision_macro', 'recall_macro', 'f1_macro']
    dict_result = cross_validate(oBestClassifier \
                                 , df_X_test.values \
                                 , df_y_test.values.ravel() \
                                 , cv=3 \
                                 , scoring=list_metric)

    # Print metric
    for metric in list_metric:
        print("{}: Score={:.2f}\t Std=(+/-){:.2f}".format(metric \
                                                          , dict_result['test_' + metric].mean() \
                                                          , dict_result['test_' + metric].std()))

    if is_displayed:
        display_classification_hist(oBestClassifier \
                                    , df_X_test \
                                    , df_y_test, dict_hist_display)
    return oBestClassifier


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def roc_display(classifier, X_test: np.array, y_test: np.array) -> None:
    classifier_name = str(type(classifier)).split('.')[-1].split('\'>')[0]

    # ---------------------------------------------------------------------------
    # predict probabilities
    # ---------------------------------------------------------------------------
    y_hat = classifier.predict_proba(X_test)

    # ---------------------------------------------------------------------------
    # Yield probabilities for the positive class
    # ---------------------------------------------------------------------------
    pos_probs = y_hat[:, 1]

    # ---------------------------------------------------------------------------
    # plot no skill roc curve
    # ---------------------------------------------------------------------------
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    # ---------------------------------------------------------------------------
    # calculate roc curve for model
    # ---------------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, pos_probs)

    # ---------------------------------------------------------------------------
    # calculate roc auc
    # ---------------------------------------------------------------------------
    roc_auc = roc_auc_score(y_test, pos_probs)
    classifier_name = classifier.__class__.__name__
    print('{}: ROC AUC {:.3f} '.format(classifier_name, roc_auc))

    # ---------------------------------------------------------------------------
    # plot model roc curve
    # ---------------------------------------------------------------------------
    plt.plot(fpr, tpr, marker='.', label=classifier_name)

    # ---------------------------------------------------------------------------
    # axis labels
    # ---------------------------------------------------------------------------
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # ---------------------------------------------------------------------------
    # show the legend
    # ---------------------------------------------------------------------------
    plt.legend()

    # ---------------------------------------------------------------------------
    # show the plot
    # ---------------------------------------------------------------------------
    plt.show()


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def pr_display(classifier, y: np.array, X_test: np.array, y_test: np.array) -> None:
    classifier_name = str(type(classifier)).split('.')[-1].split('\'>')[0]

    # ---------------------------------------------------------------------------
    # Predict probabilities
    # ---------------------------------------------------------------------------
    y_hat = classifier.predict_proba(X_test)

    # ---------------------------------------------------------------------------
    # Calculate the no skill line as the proportion of the positive class
    # ---------------------------------------------------------------------------
    no_skill = len(y[y == 1]) / len(y)
    if .9 < no_skill:
        no_skill = len(y[y == 0]) / len(y)
    no_skill = 0.5

    # ---------------------------------------------------------------------------
    # Plot the no skill precision-recall curve
    # ---------------------------------------------------------------------------
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

    # ---------------------------------------------------------------------------
    # Retrieve just the probabilities for the positive class
    # ---------------------------------------------------------------------------
    pos_probs = y_hat[:, 1]

    # ---------------------------------------------------------------------------
    # Calculate model precision-recall curve
    # ---------------------------------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)
    classifier_name = classifier.__class__.__name__
    auc_score = auc(recall, precision)
    print('{}: PR AUC {:.3f} '.format(classifier_name, auc_score))

    # ---------------------------------------------------------------------------
    # Plot precision-recall curve
    # ---------------------------------------------------------------------------
    plt.plot(recall, precision, marker='.', label=classifier_name)

    # ---------------------------------------------------------------------------
    # axis labels
    # ---------------------------------------------------------------------------
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # ---------------------------------------------------------------------------
    # show the legend
    # ---------------------------------------------------------------------------
    plt.legend()

    # ---------------------------------------------------------------------------
    # show the plot
    # ---------------------------------------------------------------------------
    plt.show()


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def confusion_matrix_display(classifier, X, y, figsize=(8, 8), cmap=plt.cm.Reds):
    y_pred = classifier.predict(X)
    labels = classifier.classes_
    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(cmap=cmap, ax=ax)
    plt.show()


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def clusters_gmm_explore(oSchool, scaler, dict_scaler \
                         , range_component=range(1, 5)):
    '''Explore the number of clusters using GMM and elbow method.
    '''
    # -------------------------------------------------------------------
    # Update list of quantitative features in order to take into account
    # FinalGrade feature while clustering
    # -------------------------------------------------------------------
    oSchool.list_feature_quantitative = oSchool.list_feature_quantitative + [oSchool.TARGET]

    # -------------------------------------------------------------------
    # Prepare dataset
    # -------------------------------------------------------------------
    oSchool._df_data_scale(scaler, dict_scaler)

    # -------------------------------------------------------------------
    # Clusterization and labels update
    # -------------------------------------------------------------------
    list_bic = list()
    list_aic = list()
    list_silhouette = list()

    X = oSchool.df_data
    for n_components in range_component:
        gmm = GaussianMixture(n_components=n_components, random_state=13)
        gmm.fit(X)
        labels = gmm.predict(X)
        list_bic.append(gmm.bic(X))
        list_aic.append(gmm.aic(X))
        list_silhouette.append(silhouette_score(X, labels))

    print(list_silhouette)
    plt.plot(range_component, list_bic, 'o-', label='BIC')
    plt.plot(range_component, list_aic, 'o-', label='AIC')
    plt.xlabel("Number of groups (n_components)")
    plt.ylabel("BIC and AIC values")
    plt.title("Optimal GMM groups")
    plt.legend()
    plt.show()

    plt.plot(range_component, list_silhouette, 'o-', label='Silhouette')
    plt.title("Optimal GMM groups")
    plt.legend()
    plt.show()


# -------------------------------------------------------------------------------


