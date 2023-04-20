# -*- coding: utf-8 -*-
"""
File:   assignment05.py
Author:  Kaifa Lu
Date:   11/16/21
Desc:   Hand design a neural network
    
"""

""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
%matplotlib inline

""" =======================  Figure Settings ========================== """

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.default'] = 'regular'
rcParams['mathtext.cal'] = 'Arial'
rcParams['legend.fontsize'] = 'medium'

""" ======================  Function definitions ========================== """

# Define Activate Function

def activate_function_1(X): # [0,1] = 1
    mm = len(X)
    if np.isscalar(X[0]):
        Y = np.zeros((mm,1))
        for i in range(mm):
            if X[i] >= 0 and X[i] <= 1:
                Y[i] = 1
            else:
                Y[i] = 0
    else:
        nn = len(X[0])
        Y = np.zeros((mm,nn))
        for i in range(mm):
            for j in range(nn):
                if X[i,j] >= 0 and X[i,j] <= 1:
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
    return Y

def activate_function_2(X): # (0,1] = 1
    mm = len(X)
    if np.isscalar(X[0]):
        Y = np.zeros((mm,1))
        for i in range(mm):
            if X[i] > 0 and X[i] <= 1:
                Y[i] = 1
            else:
                Y[i] = 0
    else:
        nn = len(X[0])
        Y = np.zeros((mm,nn))
        for i in range(mm):
            for j in range(nn):
                if X[i,j] > 0 and X[i,j] <= 1:
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
    return Y

def activate_function_3(X): # (0,1) = 1
    mm = len(X)
    if np.isscalar(X[0]):
        Y = np.zeros((mm,1))
        for i in range(mm):
            if X[i] > 0 and X[i] < 1:
                Y[i] = 1
            else:
                Y[i] = 0
    else:
        nn = len(X[0])
        Y = np.zeros((mm,nn))
        for i in range(mm):
            for j in range(nn):
                if X[i,j] > 0 and X[i,j] < 1:
                    Y[i,j] = 1
                else:
                    Y[i,j] = 0
    return Y

def activate_function_4(X):
    Y = X
    return Y

# Define Hand-designed Neural Network

def hand_neural_network(X):
    N = X.shape[0]
    # From Input Layer to Hidden Layer 1 
    weights_1 = np.array([[0.5,0],[0.25,0],[0.5,0],[np.double(1/3),0],[0.5,0],[0.5,0],[0.5,0],[0,np.double(1/3)],[0,1],[0,0.5],[0,0.5],[0,0.5]])
    bias_1 = np.array([[4.5,1.75,1.5,np.double(1/3),-1,-2,-3,np.double(5/3),2,0.5,-0.5,-1.5]])
    hidden_1_input = np.dot(X,weights_1.T) + np.dot(np.ones((N,1)),bias_1)
    hidden_1_output = activate_function_1(hidden_1_input[:,0])
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_3(hidden_1_input[:,1])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,2])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_3(hidden_1_input[:,3])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,4])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_2(hidden_1_input[:,5])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,6])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,7])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_3(hidden_1_input[:,8])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,9])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_3(hidden_1_input[:,10])),axis=1)
    hidden_1_output = np.concatenate((hidden_1_output,activate_function_1(hidden_1_input[:,11])),axis=1)
    # From Hidden Layer 1 to Hidden Layer 2
    weights_2 = np.array([[0,0,0,0,0,0.5,0,0,0,0.5,0,0.5],
                         [0,0,0,0,0,0,0.5,0,0,0,0,0.5],
                         [0,0,0,0,0.5,0,0,0.5,0.5,0.5,0.5,0.5],
                         [0.5,0,0.5,0,0,0,0,0.5,0.5,0.5,0.5,0.5],
                         [0,0.5,0,0,0,0,0,0.5,0,0,0,0]])
    bias_2 = np.array([[-0.5,-0.5,-0.5,-0.5,-0.5]])
    hidden_2_input = np.dot(hidden_1_output,weights_2.T) + np.dot(np.ones((N,1)),bias_2)
    hidden_2_output = activate_function_2(hidden_2_input)
    # From Hidden Layer 2 to Hidden Layer 3
    weights_3 = np.array([[0.5,0.5,0.5,-0.5,-0.5],[-0.4,-0.4,-0.4,-1,-1]])
    bias_3 = np.array([[0,1]])
    hidden_3_input = np.dot(hidden_2_output,weights_3.T) + np.dot(np.ones((N,1)),bias_3)
    hidden_3_output = activate_function_2(hidden_3_input)
    # From Hidden Layer 3 to Output Layer 
    weights_4 = np.array([[2,-1]])
    bias_4 = np.array([[0]])
    output = np.dot(hidden_3_output,weights_4.T) + np.dot(np.ones((N,1)),bias_4)
    Y = activate_function_4(output)
    return Y
    

""" =======================  Generate Data ======================= """

X = np.arange(-10,10 + 10 ** -6,0.25)
Y = np.arange(-10,10 + 10 ** -6,0.25)
XY = np.meshgrid(X, Y)
XY_test = np.vstack((XY[0].flatten(), XY[1].flatten())).T # XY_test.shape = 6561 * 2


""" ======================  Run the Model ========================== """

Y = hand_neural_network(XY_test)


""" ========================  Plot Results ============================== """

# Plot Results from Neural Network

fig1 = plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(1,1,1)
plt.scatter(XY_test[np.where(Y == 0)[0],0], XY_test[np.where(Y == 0)[0],1],color='blue')
plt.scatter(XY_test[np.where(Y == 1)[0],0], XY_test[np.where(Y == 1)[0],1],color='orange')
plt.scatter(XY_test[np.where(Y == -1)[0],0], XY_test[np.where(Y == -1)[0],1],color='white')
plt.xticks(np.arange(-10,10.1,2),fontsize=12)
plt.yticks(np.arange(-10,10.1,2),fontsize=12)
ax1.spines['left'].set_position('zero')
ax1.spines['bottom'].set_position('zero')
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
plt.grid()

# Visualize the whole output matrix
fig2 = plt.figure(figsize=(16,16))
ax2 = fig2.add_subplot(1,1,1)
for i in range(len(XY_test[:,0])):
    plt.text(XY_test[i,0],XY_test[i,1],str(int(Y[i,0])))
plt.xticks(np.arange(-10,10.1,2),fontsize=12)
plt.yticks(np.arange(-10,10.1,2),fontsize=12)
ax2.spines['left'].set_position('zero')
ax2.spines['bottom'].set_position('zero')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
plt.grid()
plt.show()