# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:01:24 2018

@author: alexc
"""

class Error(Exception):
    ''' Base class for exceptions in this module '''
    pass

class ParameterSettingError(Error):
    ''' Exception raised when parameters settings are incorrect
    
    Attributes:
        expr -- input expression in which the error occurred
        msg -- explanation of the error
    '''
    def __init__(self,msg):
        self.msg = msg
