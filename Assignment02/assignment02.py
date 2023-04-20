# -*- coding: utf-8 -*-
"""
File: assignment02.py
Author: Kaifa Lu
Date: 2021-10-11  
Desc: Multiple linear regression using RBF with an Elastic Net regularization
    
"""

""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from scipy.stats import norm
import math

plt.close('all') #close any open plots


""" =======================  Figure Settings ========================== """
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.default'] = 'regular'
rcParams['mathtext.cal'] = 'Arial'
rcParams['legend.fontsize'] = 'medium'


""" ======================  Function definitions ========================== """

def plotData(x1,t1,x2=None,t2=None,x3=None,t3=None,x4=None,t4=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'k') #plot estimated value using RBF with evenly space centers
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot estimated value using RBF with the training data centers
    if(x4 is not None):
        p4 = plt.plot(x4, t4, 'y') #plot true value

    #add title, legend and axes labels
    plt.ylim([-10,10])
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x2 is None):
        plt.legend((p1[0]),legend)
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    if(x4 is None):
        plt.legend((p1[0],p2[0],p3[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0],p4[0]),legend)
        
def fitdata(x,t,M,s,mu):
    # Without any regularization item
    X = np.array([[1 for j in np.arange(len(x))]]+[[math.exp(-(x[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x))] for i in np.arange(M)]).T
    w = np.linalg.inv(X.T@X+np.eye((X.T@X).shape[1])*10**-6)@X.T@t # guarantee positive definite matrix
    return w

def fitdata_regularization(x,t,M,s,mu,alpha,lamda): #given initial weight based on fitdata without any regularization
    #With Elastic Net Regularization
    learning_rate = 0.001
    iteration_max = 10000
    threshold = 1e-4
    w = fitdata(x,t,M,s,mu)
    X = np.array([[1 for j in np.arange(len(x))]]+[[math.exp(-(x[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x))] for i in np.arange(M)]).T
    err = np.linalg.norm(X@w-t)**2+alpha*(lamda*np.linalg.norm(w,ord=1)+(1-lamda)*np.linalg.norm(w)**2)
    iteration_num = 1
    delta = err
    while (delta>threshold)&(iteration_num<iteration_max):
        for i in range(len(w)):
            if i == 0:
                tt=np.sum([(X[j]@w-t[j]) for j in range (M)])
            else:
                tt=np.sum([(X[j]@w-t[j])*math.exp(-(x[j]-mu[i-1])**2/(2*s**2)) for j in range (M)])
            if w[i]<0:
                w[i]=w[i]-learning_rate*(tt-alpha*lamda+2*alpha*(1-lamda)*w[i])
            elif w[i]>0:
                w[i]=w[i]-learning_rate*(tt+alpha*lamda+2*alpha*(1-lamda)*w[i])
            else:
                w[i]=10**-6
        error = np.linalg.norm(X@w-t)**2+alpha*(lamda*np.linalg.norm(w,ord=1)+(1-lamda)*np.linalg.norm(w)**2)
        delta = abs(err-error)
        err = error
        iteration_num += 1
    return w

def plot_alpha_lamda(x,t,M,s,mu,alpha,lamda,w):
    x_range = np.arange(int(min(x))-1,int(max(x))+1,0.001)
    X_range = np.array([[1 for i in np.arange(len(x_range))]]+[[math.exp(-(x_range[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x_range))] for i in np.arange(M)]).T
    est_y = X_range@w
    real_y = np.array([3 * (x_range[i] + math.sin(x_range[i])) * math.exp(-x_range[i]**2) for i in range (len(x_range))])
    plotData(x,t,x_range,est_y,x_range,real_y,legend=['Training Data','Estimated Function','True Function'])
    plt.show()

def data_MAPE(y_pred,y_true):
    y_true[np.where(y_true==0)] = 10**-6
    mape = np.mean(np.abs((y_pred-y_true) / y_true))*100
    return mape
    
def data_matrix(x,t,M,s,mu,w):
    error = []
    X = np.array([[1 for j in np.arange(len(x))]]+[[math.exp(-(x[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x))] for i in np.arange(M)]).T
    error.append(np.linalg.norm(X@w-t,ord=1)) #Absolute error
    error.append(np.linalg.norm(X@w-t,ord=1)/len(t)) # MAE
    error.append(np.linalg.norm(X@w-t)**2/len(t)) # MSE
    error.append(data_MAPE(X@w,t)) # MAPE
    return error

def data_alpha_lamda(x1,t1,x2,t2,x3,t3,M,s,mu):
    err=[[],[],[]]
    nn=0
    for alpha in range (11):
        for lamda in np.linspace(0,1,101):
            w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda) # optimal weight
            err[0].append(data_matrix(x1,t1,M,s,mu,w))
            err[1].append(data_matrix(x2,t2,M,s,mu,w))
            err[2].append(data_matrix(x3,t3,M,s,mu,w))
            nn = nn+1
            print(nn,'alpha=',alpha,'lambda=',lamda,'Absolute error: train_err=',err[0][nn-1][0],'valid_err=',err[1][nn-1][0],'test_err=',err[2][nn-1][0])
    return err    


""" =======================  Load Training Data ======================= """
data_uniform = np.load('data_set.npz')
training_set = data_uniform['arr_0']
x1 = training_set[:,0]
t1 = training_set[:,1]


""" ======================  Variable Declaration ========================== """
M = len(x1) #regression model order, number of centers
s = 0.01
alpha = 5
lamda = 0.5

    
""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """
mu = np.sort(x1)
w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda)


""" ======================== Load Validaton Data and Select the Model Hyperparameter =========================== """

"""This is where you should load the validation data set. You shoud NOT re-train the model   """
# Load the validation set
validation_set = data_uniform['arr_1']
x2 = validation_set[:,0]
t2 = validation_set[:,1]


""" ======================== Load Test Data and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
# Load the testing set
testing_set = data_uniform['arr_2']
x3 = testing_set[:,0]
t3 = testing_set[:,1]


""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """
x_range = np.arange(int(min(x1))-1,int(max(x1))+1,0.001)
X_range = np.array([[1 for i in np.arange(len(x_range))]]+[[math.exp(-(x_range[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x_range))] for i in np.arange(M)]).T
est_y = X_range@w
real_y = np.array([3 * (x_range[i] + math.sin(x_range[i])) * math.exp(-x_range[i]**2) for i in range (len(x_range))])

plt.figure(1)
print('M=',M,'s=',s,'alpha=',alpha,'lambda1=',lamda)
print('Weight:',w)
plotData(x1,t1,x_range,est_y,x_range,real_y,legend=['Training Data','Estimated Function','True Function'])
plt.show()

alpha_range = np.arange(11)
lamda_range = np.linspace(0,1,101)
alpha_plot,lamda_plot = np.meshgrid(alpha_range,lamda_range)
err = data_alpha_lamda(x1,t1,x2,t2,x3,t3,M,s,mu)

""" ========================  Prediction Plots ============================= """
# M=50, s=0.01, lambda1=lambda2=0.5, learning rate=0.001, alpha vary from 0 to 10
lamda = 0.5
fig_num = 2
for alpha in np.arange(11):
    w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda)
    plt.figure(fig_num)
    fig_num = fig_num+1
    print('M=',M,'s=',s,'alpha=',alpha,'lambda1=',lamda)
    print('Weight:',w)
    plot_alpha_lamda(x1,t1,M,s,mu,alpha,lamda,w)

# M=50, s=0.01, alpha=1, learning rate=0.001, lambda1 vary from 0 to 1
alpha = 1
for lamda in np.linspace(0,1,11):
    w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda)
    plt.figure(fig_num)
    fig_num = fig_num+1
    print('M=',M,'s=',s,'alpha=',alpha,'lambda1=',lamda)
    print('Weight:',w)
    plot_alpha_lamda(x1,t1,M,s,mu,alpha,lamda,w)

# M=50, s=0.01, alpha=5, learning rate=0.001, lambda1 vary from 0 to 1
alpha = 5
for lamda in np.linspace(0,1,11):
    w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda)
    plt.figure(fig_num)
    fig_num = fig_num+1
    print('M=',M,'s=',s,'alpha=',alpha,'lambda1=',lamda)
    print('Weight:',w)
    plot_alpha_lamda(x1,t1,M,s,mu,alpha,lamda,w)

# M=50, s=0.01, alpha=10, learning rate=0.001, lambda1 vary from 0 to 1
alpha = 10
for lamda in np.linspace(0,1,11):
    w = fitdata_regularization(x1,t1,M,s,mu,alpha,lamda)
    plt.figure(fig_num)
    fig_num = fig_num+1
    print('M=',M,'s=',s,'alpha=',alpha,'lambda1=',lamda)
    print('Weight:',w)
    plot_alpha_lamda(x1,t1,M,s,mu,alpha,lamda,w)


""" ========================  Error Plots ============================= """
# Absolute error
fig1 = plt.figure(fig_num)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('The Absolute Error on the Training Data')
ax1 = fig1.gca(projection='3d')
surf1 = ax1.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[0])[:,0]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax1.set_xlabel(r'${\alpha}$')
ax1.set_ylabel(r'${\lambda}_{1}$')
ax1.set_zlabel('Absolute_error')
fig1.colorbar(surf1, shrink=0.5, aspect=5)
plt.show()

fig2 = plt.figure(fig_num+1)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('The Absolute Error on the Validation Data')
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[1])[:,0]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax2.set_xlabel(r'${\alpha}$')
ax2.set_ylabel(r'${\lambda}_{1}$')
ax2.set_zlabel('Absolute_error')
fig2.colorbar(surf2, shrink=0.5, aspect=5)
plt.show()

fig3 = plt.figure(fig_num+2)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('The Absolute Error on the Testing Data')
ax3 = fig3.gca(projection='3d')
surf3 = ax3.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[2])[:,0]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax3.set_xlabel(r'${\alpha}$')
ax3.set_ylabel(r'${\lambda}_{1}$')
ax3.set_zlabel('Absolute_error')
fig3.colorbar(surf3, shrink=0.5, aspect=5)
plt.show()

# MAE
fig4 = plt.figure(fig_num+3)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Error on the Training Data')
ax4 = fig4.gca(projection='3d')
surf4 = ax4.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[0])[:,1]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax4.set_xlabel(r'${\alpha}$')
ax4.set_ylabel(r'${\lambda}_{1}$')
ax4.set_zlabel('MAE')
fig4.colorbar(surf4, shrink=0.5, aspect=5)
plt.show()

fig5 = plt.figure(fig_num+4)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Error on the Validation Data')
ax5 = fig5.gca(projection='3d')
surf5 = ax5.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[1])[:,1]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax5.set_xlabel(r'${\alpha}$')
ax5.set_ylabel(r'${\lambda}_{1}$')
ax5.set_zlabel('MAE')
fig5.colorbar(surf5, shrink=0.5, aspect=5)
plt.show()

fig6 = plt.figure(fig_num+5)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Error on the Testing Data')
ax6 = fig6.gca(projection='3d')
surf6 = ax6.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[2])[:,1]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax6.set_xlabel(r'${\alpha}$')
ax6.set_ylabel(r'${\lambda}_{1}$')
ax6.set_zlabel('MAE')
fig6.colorbar(surf6, shrink=0.5, aspect=5)
plt.show()

# MSE
fig7 = plt.figure(fig_num+6)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Squared Error on the Training Data')
ax7 = fig7.gca(projection='3d')
surf7 = ax7.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[0])[:,2]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax7.set_xlabel(r'${\alpha}$')
ax7.set_ylabel(r'${\lambda}_{1}$')
ax7.set_zlabel('MSE')
fig7.colorbar(surf7, shrink=0.5, aspect=5)
plt.show()

fig8 = plt.figure(fig_num+7)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Squared Error on the Validation Data')
ax8 = fig8.gca(projection='3d')
surf8 = ax8.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[1])[:,2]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax8.set_xlabel(r'${\alpha}$')
ax8.set_ylabel(r'${\lambda}_{1}$')
ax8.set_zlabel('MSE')
fig8.colorbar(surf8, shrink=0.5, aspect=5)
plt.show()

fig9 = plt.figure(fig_num+8)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Squared Error on the Testing Data')
ax9 = fig9.gca(projection='3d')
surf9 = ax9.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[2])[:,2]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax9.set_xlabel(r'${\alpha}$')
ax9.set_ylabel(r'${\lambda}_{1}$')
ax9.set_zlabel('MSE')
fig9.colorbar(surf9, shrink=0.5, aspect=5)
plt.show()

# MAPE
fig10 = plt.figure(fig_num+9)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Percentage Error on the Training Data')
ax10 = fig10.gca(projection='3d')
surf10 = ax10.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[0])[:,3]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax10.set_xlabel(r'${\alpha}$')
ax10.set_ylabel(r'${\lambda}_{1}$')
ax10.set_zlabel('MAPE')
fig10.colorbar(surf10, shrink=0.5, aspect=5)
plt.show()

fig11 = plt.figure(fig_num+10)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Percentage Error on the Validation Data')
ax11 = fig11.gca(projection='3d')
surf11 = ax11.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[1])[:,3]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax11.set_xlabel(r'${\alpha}$')
ax11.set_ylabel(r'${\lambda}_{1}$')
ax11.set_zlabel('MAPE')
fig11.colorbar(surf11, shrink=0.5, aspect=5)
plt.show()

fig12 = plt.figure(fig_num+11)
print('M=',M,'s=',s)
print('alpha: range=','[0,10]','step_size=',1)
print('lambda: range=','[0,1]','step_size=',0.01)
print('Mean Absolute Percentage Error on the Testing Data')
ax12 = fig12.gca(projection='3d')
surf12 = ax12.plot_surface(alpha_plot,lamda_plot,np.array(np.array(err[2])[:,3]).reshape(11,101).T,cmap=cm.coolwarm,linewidth=0)
ax12.set_xlabel(r'${\alpha}$')
ax12.set_ylabel(r'${\lambda}_{1}$')
ax12.set_zlabel('MAPE')
fig12.colorbar(surf12, shrink=0.5, aspect=5)
plt.show()

""" ========================  Plot Discussion Results ============================== """
x_range = np.arange(int(min(min(x1),min(x2),min(x3)))-1,int(max(max(x1),max(x2),max(x3)))+1,0.001)
X_range = np.array([[1 for i in np.arange(len(x_range))]]+[[math.exp(-(x_range[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x_range))] for i in np.arange(M)]).T
w = fitdata_regularization(x1,t1,M,s,mu,0,0)
est_y = X_range@w
real_y = np.array([3 * (x_range[i] + math.sin(x_range[i])) * math.exp(-x_range[i]**2) for i in range (len(x_range))])
plt.figure(fig_num+12)
plotData(x1,t1,x_range,est_y,x_range,real_y,legend=['Training Data','Estimated Function','True Function'])
print('M=',M,'s=',s,'alpha=',0,'lambda1=',0)
print('Weight:',w)
plt.show()