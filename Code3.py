# Code 3 shows us how n_estimators can affect the r2 score and give us a better output


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

FVM_values1 = np.array(y_test)
h5_FVM1 = FVM_values1[:,4]
FVM_values2 = np.array(y_train)
h5_FVM2 = FVM_values2[:,4]

N_trees = 50                      # value of n_estimators in Random Forest Algorithm
r2_test5 = np.zeros(N_trees)      # Comparison of h5 for different values of n_estimators
r2_train5 = np.zeros(N_trees)

for i in range(N_trees):
    regressor = RandomForestRegressor(n_estimators = (i+1), random_state = 0)                      # creating a variable class for Random forest regressor
    regressor.fit(features_x,y)


    y_pred1 = regressor.predict(x_test)                      # predicting y values for input test matrix for random forest algorithm   
    h5_ML1 = y_pred1[:,4]
    y_pred2 = regressor.predict(x_train)                      # predicting y values for input test matrix for random forest algorithm   
    h5_ML2 = y_pred2[:,4]
   
    r2_test5[i] = r2_score(h5_FVM1, h5_ML1)    
    r2_train5[i] = r2_score(h5_FVM2, h5_ML2)  

# plot for r2 score vs n_estimators
fig, plot_points = plt.subplots()
plot_points.set_title('r2 score vs n_estimators')

plot_points.set_ylabel('r2 score')
plot_points.set_xlabel('n_estimator value')

plt.plot([i for i in range(N_trees)],r2_train5)
plt.show()