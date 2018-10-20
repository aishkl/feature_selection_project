# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    
    fs=SelectPercentile(f_regression,k)
    X_new=fs.fit_transform(X,y)    
    
    sd=list(X.columns.values[np.argsort(fs.scores_)[:-8:-1]])
    return sd

percentile_k_features(data,k=20)


