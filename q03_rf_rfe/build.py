# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(data):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    ls=[]
    rf=RandomForestClassifier()
    rfe = RFE(rf, X.shape[1]/2)
    X_new=rfe.fit(X,y)
    mask = rfe.support_ #list of booleans
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask,X.columns.values):
        if bool:
            new_features.append(feature)
    return new_features
   
        
    

rf_rfe(data)
    



