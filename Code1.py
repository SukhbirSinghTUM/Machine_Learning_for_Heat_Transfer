# Code1 is a program which computes the values of h1-h5 and plots the comparison graph between values predicted by ML and FVM models
# Size of training sample = 75% 
# The graphs in the original research paper are referenced as "Fig.3: Comparison of h predicted by FVM and ML regressor"


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


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)      # creating a variable class for Random forest regressor
regressor.fit(features_x,y)

# predicting y values for input test matrix 
y_pred1 = regressor.predict(x_test)                        
FVM_values = np.array(y_test)       # for converting list into array as TrainTestSplit returns a list and not array

# splitting the 2D array for h1-h5 values for ML model
h1_ML = y_pred1[:,0]
h2_ML = y_pred1[:,1]
h3_ML = y_pred1[:,2]
h4_ML = y_pred1[:,3]
h5_ML = y_pred1[:,4]

# splitting the 2D array for h1-h5 values for FVM model
h1_FVM = FVM_values[:,0]
h2_FVM = FVM_values[:,1]
h3_FVM = FVM_values[:,2]
h4_FVM = FVM_values[:,3]
h5_FVM = FVM_values[:,4]

# Plotting the predicted values by Random Forests Algorithm vs values obtained from FVM 

######## h1 PLOTS ########
fig, plot_points = plt.subplots()
plot_points.set_title('h1_ML vs h1_FVM')

plot_points.set_ylabel('h1,FVM (W/m2K)')
plot_points.set_xlabel('h1,ML (W/m2K)')
plot_points.plot([0,1],[0,1], transform=plot_points.transAxes)

plt.plot(h1_ML,h1_FVM, 'o')
plt.show()

######## h2 PLOTS ########
fig, plot_points = plt.subplots()
plot_points.set_title('h2_ML vs h2_FVM')

plot_points.set_ylabel('h2,FVM (W/m2K)')
plot_points.set_xlabel('h2,ML (W/m2K)')
plot_points.plot([0,1],[0,1], transform=plot_points.transAxes)

plt.plot(h2_ML,h2_FVM, 'o')
plt.show()

######## h3 PLOTS ########
fig, plot_points = plt.subplots()
plot_points.set_title('h3_ML vs h3_FVM')

plot_points.set_ylabel('h3,FVM (W/m2K)')
plot_points.set_xlabel('h3,ML (W/m2K)')
plot_points.plot([0,1],[0,1], transform=plot_points.transAxes)

plt.plot(h3_ML,h3_FVM, 'o')
plt.show()

######## h4 PLOTS ########
fig, plot_points = plt.subplots()
plot_points.set_title('h4_ML vs h4_FVM')

plot_points.set_ylabel('h4,FVM (W/m2K)')
plot_points.set_xlabel('h4,ML (W/m2K)')
plot_points.plot([0,1],[0,1], transform=plot_points.transAxes)

plt.plot(h4_ML,h4_FVM, 'o')
plt.show()

######## h5 PLOTS ########
fig, plot_points = plt.subplots()
plot_points.set_title('h5_ML vs h5_FVM')

plot_points.set_ylabel('h5,FVM (W/m2K)')
plot_points.set_xlabel('h5,ML (W/m2K)')
plot_points.plot([0,1],[0,1], transform=plot_points.transAxes)

plt.plot(h5_ML,h5_FVM, 'o')
plt.show()
