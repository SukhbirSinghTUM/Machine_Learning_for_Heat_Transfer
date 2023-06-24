#!/usr/bin/env python
# coding: utf-8

# In[49]:


# Libraries used 
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression        # Simple linear regression
from sklearn.model_selection import train_test_split     # for splitting data set into training and testing data sets
from sklearn.ensemble import RandomForestRegressor       # Random Forest Regressor import
from sklearn.metrics import r2_score,mean_squared_error  # For validation of the model
import matplotlib.pyplot as plt                          # For plotting


# In[50]:


data = pd.read_excel ('Data_FVM.xlsx')                   # Reading the entire feature matrix


# In[51]:


features_x = data[['e1','e2','e3','e4','e5']]            # splitting the feature matrix for extracting input features
y = data[['h1','h2','h3','h4','h5']]                     # extracting the output values h1-h5 
x_train, x_test, y_train, y_test = train_test_split(features_x, y, test_size=0.30, random_state=42)     # splitting the x and y matrices into training and testing data sets


# In[52]:


regressor1 = RandomForestRegressor()                      # creating a variable class for Random forest regressor
regressor1.fit(features_x,y)
regressor2 = LinearRegression()                           # creating a variable class for simple regressor
regressor2.fit(x_train,y_train)


# In[53]:


y_pred1 = regressor1.predict(x_test)                      # predicting y values for input test matrix for random forest algorithm                
y_pred2 = regressor2.predict(x_test)                      # predicting y values for input test matrix for simple linear regressor


# In[54]:


mse1 = mean_squared_error(y_test,y_pred1)                 # mean squared error for comparing the test values for RF algorithm
rmse1 = np.sqrt(mse1)
print(rmse1)
mse2 = mean_squared_error(y_test,y_pred2)                 # mean squared error for comparing the test values for simple linear regressor
rmse2 = np.sqrt(mse2)
print(rmse2)
test_score1 = r2_score(y_test, y_pred1)                    # R square value for RF algorithm
print(test_score1)
test_score2 = r2_score(y_test, y_pred2)                    # R square value for simple linear regressor
print(test_score2)


# In[55]:


# Plotting the predicted values vs values obtained from FVM 
    
fig, plot_points = plt.subplots()
plot_points.set_title('h_FVM vs h_ML (RF algorithm)')

plot_points.set_ylabel('h_ML')
plot_points.set_xlabel('h_FVM')

plt.plot(y_test,y_pred1, 'o')
plt.show()


# In[56]:


# Plotting the predicted values vs values obtained from FVM 
    
fig, plot_points = plt.subplots()
plot_points.set_title('h_FVM vs h_ML (Simple Linear Regressor)')

plot_points.set_ylabel('h_ML')
plot_points.set_xlabel('h_FVM')

plt.plot(y_test,y_pred2, 'o')
plt.show()


# In[ ]:




