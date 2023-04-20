# -*- coding: utf-8 -*-
"""
File:   assignment01.py
Author: Kaifa Lu
Date: 2021-09-20  
Desc: Multiple linear regression using RBF 
    
"""


""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

plt.close('all') #close any open plots


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
    X = np.array([[1 for j in np.arange(len(x1))]]+[[math.exp(-(x1[j]-mu[i])**2/(2*s**2)) for j in np.arange(len(x1))] for i in np.arange(M)]).T
    w = np.linalg.inv(X.T@X+np.eye((X.T@X).shape[1])*10**-6)@X.T@t
    return w

def plot_Ms(x1,t1,x2,M,s): # plot multiple figures under different M and s
    for i in np.arange(len(M)):
        for j in np.arange(len(s)):
            #Evenly spaced centers
            mu_even = np.linspace(min(x1),max(x1),M[i])
            w_even = fitdata(x1,t1,M[i],s[j],mu_even)
            X_even = np.array([[1 for mm in np.arange(len(x2))]]+[[math.exp(-(x2[nn]-mu_even[mm])**2/(2*s[j]**2)) for nn in np.arange(len(x2))] for mm in np.arange(M[i])]).T
            esty_even=X_even@w_even
            #Randomly sampled centers
            mu_random = np.sort(x1[np.random.choice(len(x1),M[i],replace=False)])
            w_random = fitdata(x1,t1,M[i],s[j],mu_random)
            X_random = np.array([[1 for mm in np.arange(len(x2))]]+[[math.exp(-(x2[nn]-mu_random[mm])**2/(2*s[j]**2)) for nn in np.arange(len(x2))] for mm in np.arange(M[i])]).T
            esty_random=X_random@w_random
            # Plot
            xrange = np.arange(int(min(min(x1),min(x2)))-1,int(max(max(x1),max(x2)))+1,0.001)
            realy = xrange / (1+xrange)
            print('M=',M[i],'s=',s[j])
            print('Evenly Spaced Centers:',w_even)
            print('Randomly Sampled Centers:',w_random)
            if len(s)>1:
                plt.figure(j+22)
            else:
                plt.figure(i+2) # if else structure is used for plotting conveniently
            plotData(x1,t1,x2,esty_even,x2,esty_random,xrange,realy,legend=['Training Data','Evenly Spaced Center','Randomly Sampled Center','True Function'])
            plt.show()


def testdata_even(x1,t1,x2,M,s): #x1,training data;x2,test data;M,s are both vectors
    err=[]
    for i in np.arange(len(M)):
        for k in np.arange(len(s)):
            mu = np.linspace(min(x1),max(x1),M[i])
            w = fitdata(x1,t1,M[i],s[k],mu)
            X = np.array([[1 for i in np.arange(len(x2))]]+[[math.exp(-(x2[j]-mu[i])**2/(2*s[k]**2)) for j in np.arange(len(x2))] for i in np.arange(M[i])]).T
            y = X@w
            t = x2/(1+x2)
            err.append(np.mean([abs(y[mm]-t[mm]) for mm in np.arange(len(x2))]))
    return err

def testdata_random(x1,t1,x2,M,s): #x1,training data;x2,test data;M,s are both vectors
    err=[]
    for i in np.arange(len(M)):
        for k in np.arange(len(s)):
            error=[]
            for j in np.arange(10): # run 10 times
                mu = np.sort(x1[np.random.choice(len(x1),M[i],replace=False)])
                w = fitdata(x1,t1,M[i],s[k],mu)
                X = np.array([[1 for i in np.arange(len(x2))]]+[[math.exp(-(x2[j]-mu[i])**2/(2*s[k]**2)) for j in np.arange(len(x2))] for i in np.arange(M[i])]).T
                y = X@w
                t = x2/(1+x2)
                error.append(np.mean([abs(y[mm]-t[mm]) for mm in np.arange(len(x2))]))
            err.append([np.mean(error),np.std(error)])
    return err

def testdata_polynomial(x1,t1,x2,M):
    err=[]
    for i in np.arange(len(M)):
        X1 = np.array([x1**m for m in range(M[i]+1)]).T
        w = np.linalg.inv(X1.T@X1)@X1.T@t1
        X2 = np.array([x2**m for m in range(M[i]+1)]).T
        y = X2@w
        t = x2/(1+x2)
        err.append(np.mean([abs(y[mm]-t[mm]) for mm in np.arange(len(x2))]))
    return err

""" ======================  Variable Declaration ========================== """
M =  3 #regression model order
s = 0.5

""" =======================  Load Training Data ======================= """
data_uniform = np.load('train_data.npy').T
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]

    
""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """
#Evenly spaced centers
mu_even = np.linspace(min(x1),max(x1),M)
w_even = fitdata(x1,t1,M,s,mu_even)

#Randomly sampled centers
mu_random = np.sort(x1[np.random.choice(len(x1),M,replace=False)])
w_random = fitdata(x1,t1,M,s,mu_random)


""" ======================== Load Test Data  and Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
# Load the testing set
x2 = np.load('test_data.npy')

# Evenly spaced centers
X_even = np.array([[1 for i in np.arange(len(x2))]]+[[math.exp(-(x2[j]-mu_even[i])**2/(2*s**2)) for j in np.arange(len(x2))] for i in np.arange(M)]).T
esty_even=X_even@w_even

# Randomly sampled centers
X_random = np.array([[1 for i in np.arange(len(x2))]]+[[math.exp(-(x2[j]-mu_random[i])**2/(2*s**2)) for j in np.arange(len(x2))] for i in np.arange(M)]).T
esty_random=X_random@w_random


""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """
xrange = np.arange(int(min(min(x1),min(x2)))-1,int(max(max(x1),max(x2)))+1,0.001)
realy = xrange / (1+xrange)

fig = plt.figure(1) # Example: M=3,s=0.5
print('M=',3,'s=',0.5)
print('Evenly Spaced Centers:',w_even)
print('Randomly Sampled Centers:',w_random)
plotData(x1,t1,x2,esty_even,x2,esty_random,xrange,realy,legend=['Training Data','Evenly Spaced Center','Randomly Sampled Center','True Function'])
plt.show()

# Plot multiple figures under different M and s
# Vary M from 1 to 20, set s=0.5
Plot_M=np.arange(1,21)
s=[0.5]
plot_Ms(x1,t1,x2,Plot_M,s)
# Vary s from 0.001 to 10, set M=5
M=[5]
Plot_s=[0.001,0.01,0.1,1,10]
plot_Ms(x1,t1,x2,M,Plot_s)


fig = plt.figure(27)
Mrange = np.arange(1,21)
s=[0.5]
err_even = testdata_even(x1,t1,x2,Mrange,s)
err_random = testdata_random(x1,t1,x2,Mrange,s)
p1 = plt.plot(Mrange, err_even, 'k-')
p2 = plt.errorbar(Mrange, np.array(err_random)[:,0], yerr=np.array(err_random)[:,1])
plt.xlabel('M')
plt.ylabel('|y-t|')
plt.legend(['Evenly Spaced Center','Randomly Sampled Center'])

fig = plt.figure(28)
M = [5]
srange=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
err_even = testdata_even(x1,t1,x2,M,srange)
err_random = testdata_random(x1,t1,x2,M,srange)
p1 = plt.plot(np.log10(srange), err_even, 'k-')
p2 = plt.errorbar(np.log10(srange), np.array(err_random)[:,0], yerr=np.array(err_random)[:,1])
plt.xlabel('lg(s)')
plt.ylabel('|y-t|')
plt.legend(['Evenly Spaced Center','Randomly Sampled Center'])

fig = plt.figure(29)
# Polynomial curve fitting and RBFs with evenly spaced centers and randomly sampled centers
Mrange = np.arange(1,21)
s=[0.5]
err_poly=testdata_polynomial(x1,t1,x2,Mrange)
err_even = testdata_even(x1,t1,x2,Mrange,s)
err_random = testdata_random(x1,t1,x2,Mrange,s)
p1 = plt.plot(Mrange, err_even, 'k-')
p2 = plt.errorbar(Mrange, np.array(err_random)[:,0], yerr=np.array(err_random)[:,1])
p3 = plt.plot(Mrange, err_poly, 'y-')
plt.xlabel('M')
plt.ylabel('|y-t|')
plt.legend(['RBFs with Evenly Spaced Center','Polynomial Basis Function','RBFs with Randomly Sampled Center'],loc='upper left')
plt.ylim([0,5])
plt.show()
