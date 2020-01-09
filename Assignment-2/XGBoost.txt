#%% Importing the minimum required libraries ( other libraries imported as we progress)
import pandas as pd
import numpy as np

# Data - preprocessing section
#%% checking for missing data
df = pd.read_csv("Churn_Modelling.csv")
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
df.head()
# here we dont have any NULL or missing values. so, ignoring this


#%% dropping unnecessary columns
df=df.drop(['RowNumber','CustomerId','Surname'],axis='columns')
df.describe()

# #%%Correlation heatmap
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# sns.heatmap(df.corr(), cmap='BuGn',annot=True)
#%% splitting data into Train, DEV, test
from sklearn.model_selection import train_test_split
y=df.Exited # pulling values into another array so that we can drop
X=df.drop(['Exited'],axis='columns')
X_train, X_Dev, y_train, y_Dev = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=0,shuffle=False)

 
########## Running operations on Train Data
#%% [Train] divide train data into categories , numerical and binary
binary_columns=["HasCrCard","IsActiveMember"]
binary_df=pd.DataFrame(X_train[binary_columns])

numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
numerical_df=pd.DataFrame(X_train[numerical_columns])

category_columns=['Geography','Gender']
category_df=pd.DataFrame(X_train[category_columns])


#%% [TRAIN] Encode Categorical Data
category_df['Geography'] = category_df['Geography'].astype('category')
category_df['Gender'] = category_df['Gender'].astype('category')
category_df_Final = pd.get_dummies(category_df)

#%% [TRAIN] feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_df_train_mean=numerical_df.mean()
numerical_df_train_std=numerical_df.std(axis=0)
numerical_df_scale =pd.DataFrame(scaler.fit_transform(numerical_df),columns=numerical_columns)

#%% [TRAIN] Concatenate Columns
X_train = pd.concat([numerical_df_scale, category_df_Final,binary_df], axis=1)

########## End of Running operations on Train Data

########## running operations on Test Data
#%% [TEST] dividing data into binary, number and category
binary_columns=["HasCrCard","IsActiveMember"]
binary_df=pd.DataFrame(X_test[binary_columns])

numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
numerical_df=pd.DataFrame(X_test[numerical_columns])

category_columns=['Geography','Gender']
category_df=pd.DataFrame(X_test[category_columns])

#%% [TEST] Encode Categorical Data
category_df['Geography'] = category_df['Geography'].astype('category')
category_df['Gender'] = category_df['Gender'].astype('category')
category_df_Final = pd.get_dummies(category_df)

#%% [TEST] feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_df["CreditScore"]=(numerical_df["CreditScore"]-numerical_df_train_mean["CreditScore"]).div(numerical_df_train_std["CreditScore"])
numerical_df["Age"]=(numerical_df["Age"]-numerical_df_train_mean["Age"]).div(numerical_df_train_std["Age"])
numerical_df["Tenure"]=(numerical_df["Tenure"]-numerical_df_train_mean["Tenure"]).div(numerical_df_train_std["Tenure"])
numerical_df["Balance"]=(numerical_df["Balance"]-numerical_df_train_mean["Balance"]).div(numerical_df_train_std["Balance"])
numerical_df["NumOfProducts"]=(numerical_df["NumOfProducts"]-numerical_df_train_mean["NumOfProducts"]).div(numerical_df_train_std["NumOfProducts"])
numerical_df["EstimatedSalary"]=(numerical_df["EstimatedSalary"]-numerical_df_train_mean["EstimatedSalary"]).div(numerical_df_train_std["EstimatedSalary"])

#%%[TEST] Concatenate Columns
X_test = pd.concat([numerical_df, category_df_Final,binary_df], axis=1)


########## End of operations on Test Data

############## End of Data Pre-processing
#%%
from xgboost import XGBClassifier
model = XGBClassifier()    
model.fit(X_train, y_train)
print(model.score(X_test, y_test)*100)
#y_pred = model.predict(X_test)
#86.07142857142858
#%%
