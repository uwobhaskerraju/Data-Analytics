#%% Importing the minimum required libraries ( other libraries imported as we progress)
import pandas as pd
import numpy as np
from sklearn.svm import SVR

#%% read file into Data Frame
df = pd.read_csv('Toronto_temp.csv')
df.describe()

#%% Removing spaces and special characters
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('(', '') for c in df.columns]
df.columns = [c.replace(')', '') for c in df.columns]
df.describe()
#%% Encoding Columns based on a group
from sklearn import preprocessing
season_encoder = preprocessing.LabelEncoder()
df.season = season_encoder.fit_transform(df.season)
#%% Finding correlation between data to drop unncessary columns
corr_matrix = df.corr()
print(corr_matrix["Max_Temp_C"].sort_values(ascending=False))
# Mean_Temp_C        1.000000
# Max_Temp_C         0.991714
# Min_Temp_C         0.989511
# Month              0.312051
# Total_Rain_mm      0.120251
# Year               0.077900
# Day                0.070265
# Total_Precip_mm    0.002725
# season            -0.296395
# Total_Snow_cm     -0.395967
# Name: Max_Temp_C, dtype: float64
# Drop the columns that have negative or value almost close to zero
#Finding covariance between data to drop unncessary columns
cov_matrix = df.cov()
print(cov_matrix["Max_Temp_C"].sort_values(ascending=False))
# Max_Temp_C         131.609259
# Mean_Temp_C        122.595897
# Min_Temp_C         113.316722
# Total_Rain_mm       47.631358
# Year                14.107539
# Month               11.113533
# Day                  6.454869
# Total_Precip_mm      5.062400
# season              -3.775693
# Total_Snow_cm      -44.769164
# Name: Max_Temp_C, dtype: float64
#%% deleting the unncessary columns and viewing the dataframe using 'describe'
df=df.drop(['Date/Time','season','Total_Snow_cm','Total_Precip_mm'],axis='columns')
df.describe()
#%% Data Cleaning - replacing NaN with mean value of respective columns
df.fillna(df.mean(), inplace=True)
#%% Spliting data into Test and Train
from sklearn.model_selection import train_test_split
y=df.Max_Temp_C # value to be predicted is generally depicted by 'y'
X=df.drop(['Max_Temp_C'],axis='columns') # values or independent variables used 
#to predict 'y' are denoted by 'X' and are called as 'features'
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=False)

#%%
sv_reg = SVR(kernel='linear')
sv_reg.fit(X_train,y_train)

#%% Predicting the output using test data
y_predict=sv_reg.predict(X_test)

#%% drawing a dataframe to compare values
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df1

#%% checking various errors
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
from sklearn.metrics import r2_score
print('R2 Score:',r2_score(y_test,y_predict))
#print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
# Mean Absolute Error: 0.4981945391801813
# Mean Squared Error: 0.4885630698025133
# Root Mean Squared Error: 0.6989728677155597
# R2 Score: 0.9964050205554725


# #%% plotting
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_predict)
# ax.plot(y_test,y_predict, 'k--', c='g')
# ax.set_xlabel('measured')
# ax.set_ylabel('predicted')
# plt.show()


#%%
