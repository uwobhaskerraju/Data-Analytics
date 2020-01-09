#%% Importing the minimum required libraries ( other libraries imported as we progress)
import pandas as pd
import numpy as np
from sklearn import linear_model

#%% read file into Data Frame
df =  pd.read_csv("Churn_Modelling.csv")

#%% dropping unnecessary columns
df=df.drop(['RowNumber','CustomerId','Surname'],axis='columns')
df.describe()

#%% One hot encoding
category_columns=['Geography','Gender']
df_processed = pd.get_dummies(df, prefix_sep="__",
                              columns=category_columns)
#%% Spliting data into Test and Train
from sklearn.model_selection import train_test_split
y=df_processed.Exited # pulling values into another array so that we can drop
X=df_processed.drop(['Exited'],axis='columns')
#to predict 'y' and 'X' are called as 'features'
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=False)

#%% Create an instance of LR class and run on training data
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

#%% Predicting the output using test data
y_predict=reg.predict(X_test)

# #%% drawing a dataframe to compare values
# df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})

#%% checking various errors
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print('R2 Score :',metrics.r2_score(y_test, y_predict))

# #%% plotting
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_train['Mean_Temp_C'],X_train['Min_Temp_C'],X_train['Max_Tem'],c='blue', marker='o', alpha=0.5)
# ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='None', alpha=0.01)
# ax.set_xlabel('Price')
# ax.set_ylabel('AdSpends')
# ax.set_zlabel('Sales')
# plt.show()

# Read the following
# Theory behind linear regression
# Data Cleaning code
# Error calculation theory
# Linear regression in Excel
# Did Linear Refression match our dataset
# try by reducing the features to find how many u need to add to get the required curve
# try to understand what polynomial is and how is it different from multivariate

# Mean Absolute Error: 0.249417981271672
# Mean Squared Error: 0.12615372644895276
# Root Mean Squared Error: 0.3551812585834911

#%%
