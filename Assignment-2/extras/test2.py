#%% We will import required librabries as we progress
import pandas as pd
import numpy as np

# Data - preprocessing section
#%% checking for missing data
df = pd.read_csv("Churn_Modelling.csv")
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
df.isnull().sum()

#%%
df=df.drop(['RowNumber'],axis='columns')
df=df.drop(['Surname'],axis='columns')
# here we dont have any NULL or missing values. so, ignoring this
#%% Look for categorial values
# import preprocessing from sklearn
from sklearn import preprocessing
# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])
df.head()

#%% feature scaling
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df1 =pd.DataFrame(scaler.fit_transform(df), columns=df.columns.values)

#%% Create correlation matrix
corr_matrix = df1.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# %%
