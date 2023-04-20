# -*- coding: utf-8 -*-
"""
File:   assignment04.py
Author:  Kaifa Lu
Date:   11/02/21
Desc:   Possibilisitic C-Means
    
"""


""" =======================  Import dependencies ========================== """
"""
Code adapted from: https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/cluster/_cmeans.py
"""
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import datasets
%matplotlib inline

plt.close('all') #close any open plots


""" ======================  Function definitions ========================== """

def cmeans0(data, u_old, c, q):
    
    # Normalization
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0))) # Normalization, cluster * number of data point
    u_old = np.fmax(u_old, np.finfo(np.float64).eps) # avoid u = 0

    uq = u_old ** q
    u1q = (1-u_old) ** q

    # Calculate cluster centers
    data = data.T # data points * features
    cntr = uq.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(uq.sum(axis=1))).T) # T: cluster * feature (feature * cluster)

    # Calculate the distance of each data point from each cluster center
    d = cdist(data, cntr) # output from cdist: data points * clusters
    d_sq = d ** 2
    d_sq = np.fmax(d_sq, np.finfo(np.float64).eps)

    # Calculate parameter eita
    Eita = np.ones((c,1))
    for i in range (c):
        Eita[i] = uq[i].dot(d_sq[:,i]) / uq[i].sum()
    
    # Objective function
    jq = (uq * d_sq.T).sum() + (u1q.sum(axis=1).T).dot(Eita) # (uq * d_sq.T) * the two array have the same dimensions

    # Update u
    Eita = (np.ones((data.shape[0],1)).dot(Eita.T)).T # expand matrix
    u = 1 / (1 + (d_sq.T / Eita) ** (1. / (q - 1)))

    return cntr, u, jq, d

def cmeans(data, c, q=2, error=1e-3, maxiter=300, init=None, seed=None):
    
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1] # number of data points
        u0 = np.random.rand(c, n)
        u0 /= np.ones((c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64) # Normalization
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jq = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjq, d] = cmeans0(data, u2, c, q)
        jq = np.hstack((jq, Jjq))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = np.trace(u.dot(u.T)) / float(u.shape[1])

    return cntr, u
      

""" ======================  Variable Declaration ========================== """
#Set or load parameters here
n_samples = 1500
n_clusters = 3
q = 2

""" =======================  Generate Data ======================= """

#Blobs
blobs, y_blobs = datasets.make_blobs(n_samples=n_samples, centers=3) # Blobs: n_samples * 2; 3 clusters, y_blobs: n_samples * 1 {0,1,2}

# Anisotropicly distributed blobs
transformation = [[ 0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(blobs, transformation)
y_aniso = y_blobs

# Different variance blobs 
X_varied, y_varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])

# Unevenly sized blobs
X_filtered = np.vstack((blobs[y_blobs == 0][:500], blobs[y_blobs == 1][:100], blobs[y_blobs == 2][:10]))
    
""" ========================  Cluster the Data ============================= """

centers, L = cmeans(blobs.T, n_clusters, q) # T corresponds to inputs of functions, need transpose
             
centers_aniso, L_aniso = cmeans(X_aniso.T, n_clusters, q)
         
centers_varied, L_varied = cmeans(X_varied.T, n_clusters, q)
             
centers_filtered, L_filtered = cmeans(X_filtered.T, n_clusters, q)


""" ========================  Plot Results ============================== """

plt.figure(figsize=(12, 12))
plt.subplot(431)                                # blobs[:, 0] feature 1; blobs[:,1] feature 2;
plt.scatter(blobs[:, 0], blobs[:, 1], c=L[0,:]) # L[0,:] memeberships of all data points belonging to the first cluster
plt.title("Membership C1 - Only Blobs")
plt.subplot(432)
plt.scatter(blobs[:, 0], blobs[:, 1], c=L[1,:]) # L[1,:] memeberships of all data points belonging to the second cluster
plt.title("Membership C2 - Only Blobs")
plt.subplot(433)
plt.scatter(blobs[:, 0], blobs[:, 1], c=L[2,:]) # L[2,:] memeberships of all data points belonging to the third cluster
plt.title("Membership C3 - Only Blobs")

plt.subplot(434)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[0,:])
plt.subplot(435)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[1,:])
plt.subplot(436)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=L_aniso[2,:])
plt.title("Anisotropicly Distributed Blobs")

plt.subplot(437)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[0,:])
plt.subplot(438)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[1,:])
plt.subplot(439)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=L_varied[2,:])
plt.title("Unequal Variance - Only Blobs")

plt.subplot(4,3,10)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[0,:])
plt.subplot(4,3,11)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[1,:])
plt.subplot(4,3,12)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=L_filtered[2,:])
plt.title("Unevenly Sized Blobs")
plt.show()