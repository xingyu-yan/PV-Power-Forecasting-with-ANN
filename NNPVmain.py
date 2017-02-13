'''
Created on December 28, 2016
@author: xingyu, at Ecole Centrale de Lille
# This programme is for PV power forecasting with ANN
# reference: Coursera Machine Learning open course (Andrew Ng)
'''

import numpy as np
import scipy.io as sio
from NNPV_package.NNPVfuncs import nnCostFunction, sigmoidGradient, randInitializeWeights, checkNNGradients, trainNN, predict
from NNPV_package.NNPVfuncs import print_results

# Part 1: Loading and visualizing data

print("Loading Data ...\n")
#X_data = pd.read_csv('myRawData_X.csv')
#y_data = pd.read_csv('myRawData_y.csv')

myData = sio.loadmat('myRawData.mat')
X_data = myData['X']
y_data = myData['y']

X_train = X_data[0:420-1,:]
print(X_train.shape)
y_train = y_data[0:420-1,:]
print(y_train.shape)
X_test = X_data[420:480-1,:]
y_test = y_data[420:480-1,:]

input_layer_size = 72   
hidden_layer_size = 20  
num_labels = 24

# Part 2: Loading parameters

mat_contents = sio.loadmat('Theta.mat')

Theta1 = mat_contents['Theta1']
print(Theta1.shape)
Theta2 = mat_contents['Theta2']
print(Theta2.shape)

nn_params1 = np.matrix(np.reshape(Theta1, Theta1.shape[0]*Theta1.shape[1], order='F')).T
nn_params2 = np.matrix(np.reshape(Theta2, Theta2.shape[0]*Theta2.shape[1], order='F')).T
nn_params = np.r_[nn_params1,nn_params2]
print(nn_params.shape)
# Part 3: Compute cost (feedforward)
lamb = 0
J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at parameters without regulariyation (loaded from Theta):',J)

# Part 4: Implement regularization
lamb = 1
J,grad = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at parameters with regularization (loaded from Theta):',J)

# Part 5: Sigmoid gradient
g = sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
print ('Evaluation sigmoid gradient\n',g,'\n')

# Part 6: Initializing parameters

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)

initial_nn_params = np.r_[np.reshape(initial_Theta1,Theta1.shape[0]*Theta1.shape[1],order='F'),np.reshape(initial_Theta2,Theta2.shape[0]*Theta2.shape[1],order='F')]

# Part 7: Implement backpropagation
checkNNGradients(0)

# Part 8: Implement regularization
lamb = 3
checkNNGradients(lamb)
debug_J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train,y_train,lamb)
print ('Cost at (fixed) debugging parameters (lambda =',lamb,'):',debug_J[0],'\n')

# Part 9: Training NN
lamb = 0.04
#Theta = cgbt(initial_nn_params,X,y,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,500,1e-8)
Theta = trainNN(initial_nn_params,X_train,y_train,input_layer_size,hidden_layer_size,num_labels,lamb,0.25,0.5,50,1e-8)

Theta1 = np.matrix(np.reshape(Theta[:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1),order='F'))
Theta2 = np.matrix(np.reshape(Theta[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1),order='F'))

# Part 10: Visualize weights
#displayData(Theta1[:,1:],'courseraEx04_fig02.png')

# Part 11: Implement predict
p = predict(Theta1,Theta2,X_train).T
print_results('train_set', p-y_train)
p = predict(Theta1,Theta2,X_test).T
print_results('test_set', p-y_test)