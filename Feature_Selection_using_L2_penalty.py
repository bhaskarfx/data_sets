import numpy as np # linear algebra
import pandas as pd 
import scipy.stats as ss
from collections import Counter
import math 
from scipy import stats

url="https://raw.githubusercontent.com/bhaskarfx/data_sets/master/FIFA19_player.csv"
player_df = pd.read_csv(url, error_bad_lines=False, warn_bad_lines=False)

y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']

feature_name = list(X.columns)
num_feats=30

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


embeded_lr_feature
