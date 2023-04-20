# -*- coding: utf-8 -*-

""" =======================  Import dependencies ========================== """
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline

import math

plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """

def data_division(X,PercentageOfTrain,PercentageOfValid):
    N = len(X)
    training_set = X[0:int(PercentageOfTrain*N)]
    validation_set = X[int(PercentageOfTrain*N):int((PercentageOfTrain+PercentageOfValid)*N)]
    testing_set = X[int((PercentageOfTrain+PercentageOfValid)*N):]
    return training_set, validation_set, testing_set

def data_division_rp(X,PercentageOfTrain,PercentageOfValid):
    N = len(X)
    rp = np.random.permutation(N)
    training_set = X[rp[0:int(PercentageOfTrain*N)]]
    validation_set = X[rp[int(PercentageOfTrain*N):int((PercentageOfTrain+PercentageOfValid)*N)]]
    testing_set = X[rp[int((PercentageOfTrain+PercentageOfValid)*N):]]
    return training_set, validation_set, testing_set

def Full_covariance(X,Means,k,d,N,pZ_X):
    xDiff = X - Means[k,:] 
    J = np.zeros((d,d))
    for i in range(N):
        J = J + pZ_X[i,k]*np.outer(xDiff[i,:],xDiff[i,:])
    Sigs = J / pZ_X[:,k].sum()
    return Sigs

def Diagonal_covariance(X,Means,k,d,N,pZ_X):
    xDiff = X - Means[k,:]
    J = np.zeros((d,d))
    for i in range(len(J)):
        for j in range(N):
            J[i,i] = J[i,i] + pZ_X[j,k]*xDiff[j,i]**2
    Sigs = J / pZ_X[:,k].sum()
    return Sigs
    
def Isotropic_diagonal_covariance(X,Means,k,d,N,pZ_X):
    xDiff = X - Means[k,:]
    J = np.eye(d)
    aa = 0
    for i in range(N):
        aa = aa + pZ_X[i,k]*np.linalg.norm(xDiff[i])**2
    Sigs = J * aa / (d * pZ_X[:,k].sum())
    return Sigs

def EM_GaussianMixture(X, NumberOfComponents, Covariance_type):
    MaximumNumberOfIterations = 100
    DiffThresh = 1e-4
    N = X.shape[0] 
    d = X.shape[1]
    # Means = X[0:NumberOfComponents,:] 
    rp = np.random.permutation(N) 
    #Initialize Parameters
    Means = X[rp[0:NumberOfComponents],:] 
    Sigs = np.zeros((d,d,NumberOfComponents))
    Ps = np.zeros((NumberOfComponents)) 
    pZ_X = np.zeros((N,NumberOfComponents)) 

    for i in range(NumberOfComponents):
        Sigs[:,:,i] = np.eye(d) 
        Ps[i] = 1/NumberOfComponents 
        pZ_X[:,i] = stats.multivariate_normal.pdf(X, Means[i,:],Sigs[:,:,i])*Ps[i] 
    pZ_X = pZ_X / pZ_X.sum(axis=1)[:,np.newaxis]
    Diff = np.inf
    NumberIterations = 1
    plot_names = []
    while Diff > DiffThresh and NumberIterations <= MaximumNumberOfIterations:
        MeansOld = Means.copy()
        SigsOld = Sigs.copy()
        PsOld = Ps.copy()
        for k in range(NumberOfComponents):
            Means[k,:] = np.sum(X*pZ_X[:,k,np.newaxis],axis=0)/pZ_X[:,k].sum()
            if Covariance_type ==  'Full covariance':
                Sigs[:,:,k] = Full_covariance(X,Means,k,d,N,pZ_X)
            elif Covariance_type ==  'Diagonal covariance':
                Sigs[:,:,k] = Diagonal_covariance(X,Means,k,d,N,pZ_X)
            elif Covariance_type ==  'Isotropic diagonal covariance':
                Sigs[:,:,k] = Isotropic_diagonal_covariance(X,Means,k,d,N,pZ_X)
            else:
                print('Input wrong covariance type, check it again')
            Ps[k] = pZ_X[:,k].sum()/N
        for k in range(NumberOfComponents):
            Means[k,:][np.where(pd.isna(Means[k,:]))] = 0
            Sigs[:,:,k][np.where(pd.isna(Sigs[:,:,k]))] = 0
            pZ_X[:,k] = stats.multivariate_normal.pdf(X,Means[k,:],Sigs[:,:,k],allow_singular=True)*Ps[k]
        pZ_X = pZ_X / pZ_X.sum(axis=1)[:,np.newaxis]  
        Diff = abs(MeansOld - Means).sum() + abs(SigsOld - Sigs).sum() + abs(PsOld - Ps).sum();
        NumberIterations = NumberIterations + 1
    return Means, Sigs, Ps, pZ_X

def probabilistic_generative(classX,NumberOfComponents,X,Covariance_type):
    mu,cov,ps,pz_x = EM_GaussianMixture(classX, NumberOfComponents, Covariance_type)
    pX_Z = np.zeros((X.shape[0],NumberOfComponents))
    for k in range(NumberOfComponents):
        pX_Z[:,k] = stats.multivariate_normal.pdf(X, mean=mu[k,:], cov=cov[:,:,k], allow_singular=True)
    y = pX_Z@ps.T
    return y

def probabilistic_generative_classifier(class1X,class2X,class3X,NumberOfComponents,X,Covariance_type):
    pC1 = class1X.shape[0]/(class1X.shape[0] + class2X.shape[0] + class3X.shape[0])
    pC2 = class2X.shape[0]/(class1X.shape[0] + class2X.shape[0] + class3X.shape[0])
    pC3 = class3X.shape[0]/(class1X.shape[0] + class2X.shape[0] + class3X.shape[0])
    y1 = probabilistic_generative(class1X,NumberOfComponents,X,Covariance_type)
    y2 = probabilistic_generative(class2X,NumberOfComponents,X,Covariance_type)
    y3 = probabilistic_generative(class3X,NumberOfComponents,X,Covariance_type)
    pos1 = (y1*pC1)/(y1*pC1 + y2*pC2 + y3*pC3 )
    pos2 = (y2*pC2)/(y1*pC1 + y2*pC2 + y3*pC3 )
    pos3 = (y3*pC3)/(y1*pC1 + y2*pC2 + y3*pC3 )
    Class = []
    for i in range (len(pos1)):
        if pos1[i] == max(pos1[i],pos2[i],pos3[i]):
            Class.append('Iris-setosa')
        elif pos2[i] == max(pos1[i],pos2[i],pos3[i]):
            Class.append('Iris-versicolor')
        else:
            Class.append('Iris-virginica')
    Class = np.array(Class)
    return Class

def classifier_confusion_matrix(Class, t):
    class_name = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    confusion_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            kk = np.where(t==class_name[i])
            mm = 0
            for k in range(len(kk[0])):
                if Class[kk[0][k]] == class_name[j]:
                    mm = mm+1
            confusion_matrix[i,j] = mm
    return confusion_matrix

""" ======================  Variable Declaration ========================== """

NumberOfComponents = 3
PercentageOfTrain = 0.8
PercentageOfValid = 0.1
MaximumNumberOfIterations = 100
DiffThresh = 1e-4
# rp = np.random.permutation(N)
# Means = X[rp[0:NumberOfComponents],:] 
# Update Covariance [full coveriance, diagonal covariance, isotropic diagonal covariance]


""" =======================  Load Training Data ======================= """

data_uniform = np.array(pd.read_csv('iris.data',header=None, names=['sepal length','sepal width','petal length','petal width','class']))
Class1 = data_uniform[np.where(data_uniform[:,4]=='Iris-setosa')]
Class2 = data_uniform[np.where(data_uniform[:,4]=='Iris-versicolor')]
Class3 = data_uniform[np.where(data_uniform[:,4]=='Iris-virginica')]

# Divide data set randomly, not by class
training_set,validation_set,testing_set = data_division_rp(data_uniform,PercentageOfTrain,PercentageOfValid)
class1_train = training_set[np.where(training_set[:,4]=='Iris-setosa')]
class2_train = training_set[np.where(training_set[:,4]=='Iris-versicolor')]
class3_train = training_set[np.where(training_set[:,4]=='Iris-virginica')]

# Divide data set by class
Class1_train,Class1_valid,Class1_test = data_division_rp(Class1,PercentageOfTrain,PercentageOfValid)
Class2_train,Class2_valid,Class2_test = data_division_rp(Class2,PercentageOfTrain,PercentageOfValid)
Class3_train,Class3_valid,Class3_test = data_division_rp(Class3,PercentageOfTrain,PercentageOfValid)

x1_1 = Class1_train[:,0:4]
t1_1 = Class1_train[:,4]
x1_2 = Class2_train[:,0:4]
t1_2 = Class2_train[:,4]
x1_3 = Class3_train[:,0:4]
t1_3 = Class3_train[:,4]

x1_123 = np.vstack((x1_1,x1_2,x1_3))
t1_123 = np.array(list(t1_1) + list(t1_2) + list(t1_3))


""" ======================== Load Validaton Data =========================== """

# Divide data set by class
x2_1 = Class1_valid[:,0:4]
t2_1 = Class1_valid[:,4]
x2_2 = Class2_valid[:,0:4]
t2_2 = Class2_valid[:,4]
x2_3 = Class3_valid[:,0:4]
t2_3 = Class3_valid[:,4]
x2_123 = np.vstack((x2_1,x2_2,x2_3))
t2_123 = np.array(list(t2_1) + list(t2_2) + list(t2_3))


""" ======================== Load Test Data =========================== """

# Divide data set by class
x3_1 = Class1_test[:,0:4]
t3_1 = Class1_test[:,4]
x3_2 = Class2_test[:,0:4]
t3_2 = Class2_test[:,4]
x3_3 = Class3_test[:,0:4]
t3_3 = Class3_test[:,4]

x3_123 = np.vstack((x3_1,x3_2,x3_3))
t3_123 = np.array(list(t3_1) + list(t3_2) + list(t3_3))

""" ========================  Run the Model and Output Confusion Matrix ============================= """

covariance_type = ['Full covariance','Diagonal covariance','Isotropic diagonal covariance']
class_name = ['Iris-setosa','Iris-versicolor','Iris-virginica']
NumberOfComponents = np.arange(1,11)

Train_CM = [] # Order: for j in range(1,11) -- for i in range(covariance_type)
Valid_CM = []
Test_CM = []
Train_CM_class = [] # Order: for j in range(1,11) -- for i in range(covariance_type)
Valid_CM_class = []
Test_CM_class = []
for j in range(len(NumberOfComponents)):
    print('NumberOfComponents=:', NumberOfComponents[j])
    for i in range(len(covariance_type)):
        # Data samples randomly, not by class
        class_train = probabilistic_generative_classifier(class1_train[:,0:4], class2_train[:,0:4], class3_train[:,0:4], NumberOfComponents[j], training_set[:,0:4], covariance_type[i])
        confusion_matrix_train = classifier_confusion_matrix(class_train, training_set[:,4])
        Train_CM.append(confusion_matrix_train)
        print('confusion matrix on the training set:',covariance_type[i])
        print(confusion_matrix_train)

    for i in range(len(covariance_type)):
        class_valid = probabilistic_generative_classifier(class1_train[:,0:4], class2_train[:,0:4], class3_train[:,0:4], NumberOfComponents[j], validation_set[:,0:4], covariance_type[i])
        confusion_matrix_valid = classifier_confusion_matrix(class_valid, validation_set[:,4])
        Valid_CM.append(confusion_matrix_valid)
        print('confusion matrix on the validation set:',covariance_type[i])
        print(confusion_matrix_valid)

    for i in range(len(covariance_type)):
        class_test = probabilistic_generative_classifier(class1_train[:,0:4],class2_train[:,0:4],class3_train[:,0:4],NumberOfComponents[j], testing_set[:,0:4], covariance_type[i])
        confusion_matrix_test = classifier_confusion_matrix(class_test, testing_set[:,4])
        Test_CM.append(confusion_matrix_test)
        print('confusion matrix on the testing set:',covariance_type[i])
        print(confusion_matrix_test)

    # Data samples by class
    for i in range(len(covariance_type)):
        Class_train = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[j],x1_123,covariance_type[i])
        Confusion_matrix_train = classifier_confusion_matrix(Class_train, t1_123)
        Train_CM_class.append(Confusion_matrix_train)
        print('Confusion matrix on the training set by class:',covariance_type[i])
        print(Confusion_matrix_train)
    
    for i in range(len(covariance_type)):
        Class_valid = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[j],x2_123,covariance_type[i])
        Confusion_matrix_valid = classifier_confusion_matrix(Class_valid, t2_123)
        Valid_CM_class.append(Confusion_matrix_valid)
        print('Confusion matrix on the validation set by class:',covariance_type[i])
        print(Confusion_matrix_valid)
    
    for i in range(len(covariance_type)):
        Class_test = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[j],x3_123,covariance_type[i])
        Confusion_matrix_test = classifier_confusion_matrix(Class_test, t3_123)
        Test_CM_class.append(Confusion_matrix_test)
        print('Confusion matrix on the testing set by class:',covariance_type[i])
        print(Confusion_matrix_test)
        
# Data storage
Train_CM_dataframe = pd.DataFrame({covariance_type[i]:[Train_CM[3*j+i] for j in range(10)] for i in range(len(covariance_type))})
Valid_CM_dataframe = pd.DataFrame({covariance_type[i]:[Valid_CM[3*j+i] for j in range(10)] for i in range(len(covariance_type))})
Test_CM_dataframe = pd.DataFrame({covariance_type[i]:[Test_CM[3*j+i] for j in range(10)] for i in range(len(covariance_type))})

Train_CM_class_dataframe = pd.DataFrame({covariance_type[i]:[Train_CM_class[3*j+i] for j in range(10)] for i in range(len(covariance_type))})
Valid_CM_class_dataframe = pd.DataFrame({covariance_type[i]:[Valid_CM_class[3*j+i] for j in range(10)] for i in range(len(covariance_type))})
Test_CM_class_dataframe = pd.DataFrame({covariance_type[i]:[Test_CM_class[3*j+i] for j in range(10)] for i in range(len(covariance_type))})

print('Train_CM_dataframe:',Train_CM_dataframe)
print('Valid_CM_dataframe:',Valid_CM_dataframe)
print('Test_CM_dataframe:',Test_CM_dataframe)

print('Train_CM_class_dataframe:',Train_CM_class_dataframe)
print('Valid_CM_class_dataframe:',Valid_CM_class_dataframe)
print('Test_CM__class_dataframe:',Test_CM_class_dataframe)

""" ========================  Plot the Results ============================= """
mm = np.zeros((6,len(covariance_type),len(NumberOfComponents))) # store accuracy
for i in range (len(NumberOfComponents)):
    for k in range (len(covariance_type)):
        for j in range (len(class_name)):
            mm[0,k,i] = mm[0,k,i] + Train_CM_dataframe[covariance_type[k]][i][j][j]
            mm[1,k,i] = mm[1,k,i] + Valid_CM_dataframe[covariance_type[k]][i][j][j]
            mm[2,k,i] = mm[2,k,i] + Test_CM_dataframe[covariance_type[k]][i][j][j]
            mm[3,k,i] = mm[3,k,i] + Train_CM_class_dataframe[covariance_type[k]][i][j][j]
            mm[4,k,i] = mm[4,k,i] + Valid_CM_class_dataframe[covariance_type[k]][i][j][j]
            mm[5,k,i] = mm[5,k,i] + Test_CM_class_dataframe[covariance_type[k]][i][j][j]
        mm[0,k,i] = mm[0,k,i]/120 # Training data set
        mm[1,k,i] = mm[1,k,i]/15 # Validation data set
        mm[2,k,i] = mm[2,k,i]/15 # Testing data set
        mm[3,k,i] = mm[3,k,i]/120 # Training data set
        mm[4,k,i] = mm[4,k,i]/15 # Validation data set
        mm[5,k,i] = mm[5,k,i]/15 # Testing data set
color = ['k-','r-','y-']
for i in range (6):
    for j in range(len(covariance_type)):
        plt.plot(NumberOfComponents,mm[i][j],color[j])
    plt.legend(covariance_type)
    plt.xlabel('Number of Gaussian Components')
    plt.ylabel('Accuracy')
    plt.show()

""" ========================  Set the Optimal Parameters and Output Confusion Matrix on Three Data Sets ============================= """

print('covariance type:', covariance_type[0])
print('NumberOfComponents:',NumberOfComponents[1])

Class_train = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[1],x1_123,covariance_type[0])
Confusion_matrix_train = classifier_confusion_matrix(Class_train, t1_123)
print('Confusion matrix on the training set by class:')
print(Confusion_matrix_train)

Class_valid = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[1],x2_123,covariance_type[0])
Confusion_matrix_valid = classifier_confusion_matrix(Class_valid, t2_123)
print('Confusion matrix on the validation set by class:')
print(Confusion_matrix_valid)
    
Class_test = probabilistic_generative_classifier(x1_1,x1_2,x1_3,NumberOfComponents[1],x3_123,covariance_type[0])
Confusion_matrix_test = classifier_confusion_matrix(Class_test, t3_123)
print('Confusion matrix on the testing set by class:')
print(Confusion_matrix_test)