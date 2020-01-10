
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
from sklearn.linear_model import Ridge
rr = Ridge(alpha=1) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
embeded_rr_selector = SelectFromModel(rr, max_features=num_feats)
embeded_rr_selector.fit(X, y)

embeded_rr_support = embeded_rr_selector.get_support()
embeded_rr_feature = X.loc[:,embeded_rr_support].columns.tolist()
print(str(len(embeded_rr_feature)), 'selected features')

embeded_rr_feature
