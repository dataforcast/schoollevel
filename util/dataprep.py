# -*- coding: utf-8 -*-

'''This module contains functions for data-preparation'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from core.school import School


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_get_outliers(oSchool:School\
                    , df_X\
                    , contamination=0.1\
                    , method='isolationforest'\
                    , outliername='OUTLIERS'\
                    , perplexity=20\
                    , is_displayed=True)->pd.DataFrame :
    '''Returns a dataframe with outliers. Outliers are detected with IsolationForest 
    algorithm by default.'''
    
    oIsolationForest = IsolationForest(contamination=contamination, random_state=13)
    X = df_X.values

    oIsolationForest.fit(X)
    y_pred = oIsolationForest.predict(X)
    df_outlier = pd.DataFrame(data=y_pred, index=df_X.index\
    , columns=[outliername])
    index_outlier = df_outlier[df_outlier[outliername]==-1].index
    
    
    print("Nb of outliers={}".format(len(index_outlier)))

    index_inlier = [index for index\
     in oSchool.df_target.index if index not in index_outlier]
    
    if is_displayed :
        list_color = [1]*(len(index_inlier)+len(index_outlier))
        print(len(list_color))
        try:
            for index in index_outlier :
                list_color[index] =0
        except Exception as exception :
            print("***Outlier index={} : {}".format(index, exception))
        
        #for perplexity in range(25, 30, 5) :
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=list_color\
        , cmap='gray', edgecolors='k', s=60)

        legend1 = plt.legend(*scatter.legend_elements()\
        , title="Outliers", loc="best")
        plt.gca().add_artist(legend1)

        plt.title("t-SNE - with perplexity= {}".format(perplexity))
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()
    df_mask = df_outlier[outliername]==-1
    return df_outlier[df_mask]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_get_outliers_zscore(df:pd.DataFrame, threshold:int=3, target='StudentID') ->pd.DataFrame:
    '''Yield outliers for each column and store them in a dataframe 
    that is returned.
    Default method is the Z-score.
    This function detects outliers for each feature.
    INPUT:
        * df : dataframe from which outliers are extracted
        * threshold : the number of standard deviations
    OUTPUT :
        * daframe containing outliers with same index then input dattaframe.
        All columns wher all values are undefined are dropped. 
    '''
    df_outlier = pd.DataFrame()

    for feature_quant in df.columns :
        ser_data = df[feature_quant]
        data = ser_data.values
        mean = np.mean(data)
        std_dev = np.std(data)

        arr_z_score = (data - mean) / std_dev

        threshold = 3

        arr_outlier = data[np.abs(arr_z_score) > threshold]
        if 0 < len(arr_outlier) :
            min_outlier = np.min(arr_outlier)
            index_outlier = df[df[feature_quant]>=min_outlier].index
            df_outlier = pd.concat((df_outlier\
                                    , df.loc[index_outlier][feature_quant].reset_index())\
                                    , axis=0\
                                    , ignore_index= True)
    df_outlier.set_index(target, inplace=True)
        
    return df_outlier.dropna(axis=1, how='all', inplace=False)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def free_outlier_correlation(df:pd.DataFrame\
                             , df_outlier:pd.DataFrame\
                             , feature:str\
                             , target:str='FinalGrade'\
                             , method:str='pearson') :
    '''Remove outliers from a given dataframe and compute \
    correlations in-between features.'''
    if feature in df_outlier.columns :
        df_outlier_feature = df_outlier[feature].dropna()
        index_outlier = df_outlier_feature.index
        index_free_outlier = [index for index in df_feature_quant.index\
         if index not in index_outlier]
        df_free_outlier = df.loc[index_free_outlier][[feature, target]]
        df_free_outlier.dropna(axis=1, how='all', inplace=True)
        if feature in df_free_outlier.columns:
            df_corr_matrix = df_free_outlier.corr(method=method)
            corrcoef = round(df_corr_matrix[feature][target],2)
            print("{} -->({},{}) :\t{}".format(method, target, feature, corrcoef))
        else:
            pass
    else :
        pass
        #print('')
        #print("No outliers for feature={}\n".format(feature))
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_get_dummy_feature(df:pd.DataFrame) ->pd.DataFrame:
    # Get a dense matrix
    oOneHotEncoder = OneHotEncoder(sparse=False)
    df_dummy = pd.DataFrame()
    for feature in df.columns :
        arr_feature_encoded = oOneHotEncoder.fit_transform(df[[feature]])
        df_tmp = pd.DataFrame(  arr_feature_encoded\
                              , columns=oOneHotEncoder.categories_[0]\
                              , index = df.index)
        df_tmp.drop(df_tmp.columns[-1], axis=1, inplace=True)
        dict_col = {col:feature+'_'+col for col in df_tmp.columns}
        df_tmp.rename(columns=dict_col, inplace=True)
        df_dummy = pd.concat((df_dummy, df_tmp), axis=1)
    return df_dummy
#-------------------------------------------------------------------------------
    
