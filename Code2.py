# Code 2 is a program which calculates the r2 scores for the training sets and testing sets for every h (heat transfer coefficient)

# Libraries used 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split     # for splitting data set into training and testing data sets
from sklearn.ensemble import RandomForestRegressor       # Random Forest Regressor import
from sklearn.metrics import r2_score,mean_squared_error  # For validation of the model
import matplotlib.pyplot as plt                          # For plotting


data = pd.read_excel ('Data_FVM.xlsx')                   # Reading the entire feature matrix


features_x = data[['e1','e2','e3','e4','e5']]            # splitting the feature matrix for extracting input features
y = data[['h1','h2','h3','h4','h5']]                     # extracting the output values h1-h5 
x_train, x_test, y_train, y_test = train_test_split(features_x, y, test_size=0.25, random_state=42)     # splitting the x and y matrices into training and testing data sets


regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)                      # creating a variable class for Random forest regressor
regressor.fit(features_x,y)

###### FOR r2 scores of Testing set ######

y_pred1 = regressor.predict(x_test)                      # predicting y values for input test matrix for random forest algorithm   
FVM_values1 = np.array(y_test)

h1_ML1 = y_pred1[:,0]
h2_ML1 = y_pred1[:,1]
h3_ML1 = y_pred1[:,2]
h4_ML1 = y_pred1[:,3]
h5_ML1 = y_pred1[:,4]

h1_FVM1 = FVM_values1[:,0]
h2_FVM1 = FVM_values1[:,1]
h3_FVM1 = FVM_values1[:,2]
h4_FVM1 = FVM_values1[:,3]
h5_FVM1 = FVM_values1[:,4]

r2_test = np.zeros(5)                     # list containing r2 scores (testing set) of h1-h5
r2_test[0] = r2_score(h1_FVM1, h1_ML1)      
r2_test[1] = r2_score(h2_FVM1, h2_ML1)       
r2_test[2] = r2_score(h3_FVM1, h3_ML1)   
r2_test[3] = r2_score(h4_FVM1, h4_ML1)     
r2_test[4] = r2_score(h5_FVM1, h5_ML1)    


###### FOR r2 scores of Training set ######

y_pred2 = regressor.predict(x_train)                      # predicting y values for input test matrix for random forest algorithm   
FVM_values2 = np.array(y_train)

h1_ML2 = y_pred2[:,0]
h2_ML2 = y_pred2[:,1]
h3_ML2 = y_pred2[:,2]
h4_ML2 = y_pred2[:,3]
h5_ML2 = y_pred2[:,4]

h1_FVM2 = FVM_values2[:,0]
h2_FVM2 = FVM_values2[:,1]
h3_FVM2 = FVM_values2[:,2]
h4_FVM2 = FVM_values2[:,3]
h5_FVM2 = FVM_values2[:,4]

r2_train = np.zeros(5)                               ## list containing r2 scores (training set) of h1-h5
r2_train[0] = r2_score(h1_FVM2, h1_ML2)      
r2_train[1] = r2_score(h2_FVM2, h2_ML2)       
r2_train[2] = r2_score(h3_FVM2, h3_ML2)   
r2_train[3] = r2_score(h4_FVM2, h4_ML2)     
r2_train[4] = r2_score(h5_FVM2, h5_ML2)  

print(r2_train)
print(r2_test)

