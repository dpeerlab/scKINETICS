import numpy as np
import scipy
import os

def check_is_fitted(estimator,attr,msg=None):
    
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")
    
    if not hasattr(estimator, attr):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

    
    
        
        
        
        
    
       
