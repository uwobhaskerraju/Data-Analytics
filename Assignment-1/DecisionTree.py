#%% Importing the minimum required libraries ( other libraries imported as we progress)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

#%% read file into Data Frame
df = pd.read_csv('Toronto_temp.csv')

#%% Removing spaces and special characters
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('(', '') for c in df.columns]
df.columns = [c.replace(')', '') for c in df.columns]
df.describe()
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

#%% Create an instance of DTR class and run on training data
reg =  DecisionTreeRegressor(random_state = 0)
reg.fit(X_train,y_train)

#%% Predicting the output using test data
y_predict=reg.predict(X_test)

#%% drawing a dataframe to compare values
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})

#%% checking various errors
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

#%% Exporting Tree
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(reg, out_file ='tree.dot',feature_names=X_test.columns) 

# #%% Plotting
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_predict)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.set_xlabel('measured')
# ax.set_ylabel('predicted')
# plt.show()
# Mean Absolute Error: 0.7028571137611878
# Mean Squared Error: 0.8768246439126394
# Root Mean Squared Error: 0.9363891519622808
