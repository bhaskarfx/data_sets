import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
# no of maximum features we need to select
num_feats=30

print(feature_name)




#Pearson correlation
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))
                         [-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in
                   feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')

cor_feature
