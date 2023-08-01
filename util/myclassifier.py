# -*- coding: utf-8 -*-

'''This module contains classifiers redefinition
'''
from xgboost import XGBClassifier
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class MyXGBClassifier(XGBClassifier) :
    '''This class embbeds XGBClassifier class '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def predict_proba(self, X=None, ntree_limit=None, validate_features=False, base_margin=None) :
        return self.predict_proba(X, ntree_limit, validate_features, base_margin)
#-------------------------------------------------------------------------------

