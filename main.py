#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:29:03 2019

@author: gionuno
"""

import numpy as np;
import scipy.spatial as spt;
import numpy.random as rd;

import matplotlib.pyplot as plt;
import matplotlib.image  as img;
import matplotlib.gridspec as gs;

def lerp(L,x):
    i = int(x[0]);
    j = int(x[1]);
    u = x[0]-i;
    v = x[1]-j;
    r = (1-u)*(1-v)*L[i,j];
    if j+1 < L.shape[1]:
        r += (1-u)*v*L[i,j+1];
    if i+1 < L.shape[0]:
        r += (1-v)*u*L[i+1,j];
    if i+1 < L.shape[0] and j+1 < L.shape[1]:
        r += u*v*L[i+1,j+1];
    return r;

def rejection_sample(rho,N,c=1e-2):
    I = rho.shape[0];
    J = rho.shape[1];
    
    X = rd.rand(N,2);
    Mg = c/(I*J);
    for n in range(N):
        X[n,0] = (I-0.5)*rd.rand()+0.5;
        X[n,1] = (J-0.5)*rd.rand()+0.5;        
        rho_x = lerp(rho,X[n,:]);
        
        u = rd.rand();
        t = 0;
        while Mg*u > rho_x:
            #print(t);
            u = rd.rand();
            X[n,0] = (I-0.5)*rd.rand()+0.5;
            X[n,1] = (J-0.5)*rd.rand()+0.5;        
            rho_x = lerp(rho,X[n,:]);
            t += 1;
    return np.c_[X[:,0],X[:,1]];

def lloyd_relaxation(rho,X,T):
    Y = 1.0*np.array([[i,j] for i in range(rho.shape[0]) for j in range(rho.shape[1])]);
    for t in range(T):
        kdT = spt.cKDTree(X);
        pY = np.zeros(X.shape);
        p  = 1e-14*np.ones((X.shape[0],2));
        print(str(t)+" query");
        _,A = kdT.query(Y,k=1);
        A = A.reshape(rho.shape);
        print(str(t)+" update");
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if i % 10 == 0 and j % 10 == 0:
                    print(t,i,j)
                a = A[i,j];
                rho_y = rho[i,j];
                pY[a,0] += rho_y*i;
                pY[a,1] += rho_y*j;
                p[a]    += rho_y;
        X = pY/p;
        print(str(t)+" done");
    return X;
        
I = img.imread("cuttlefish.jpeg")/255.0;
L = np.sqrt(0.299*I[:,:,0]**2 + 0.587*I[:,:,1]**2 + 0.114*I[:,:,2]**2);

H = np.copy(L);
H /= np.sum(H);
H[H < 1e-6] = 0.0;
H /= np.sum(H);

X = rejection_sample(H,2000,1e2);
Y = lloyd_relaxation(H,X,4);

f,ax = plt.subplots(1,3);
ax[0].imshow(I);
ax[0].set_axis_off();
ax[1].imshow(H,cmap='gray');
ax[1].scatter(X[:,1],X[:,0],s=1);
ax[1].set_axis_off();
ax[2].imshow(H,cmap='gray');
ax[2].scatter(Y[:,1],Y[:,0],s=1);
ax[2].set_axis_off();
plt.show();

S = 1.0-img.imread("star.png")[:,:,0];
J = np.zeros((2*I.shape[0],2*I.shape[1],3));
for i in range(X.shape[0]):
    Ix = lerp(I,Y[i,:]);
    x = Y[i,0];
    y = Y[i,1];
    for s in range(16):
        for t in range(16):
            if S[s,t] < 0.0275:
                continue;
            u = int(2*x+s-8);
            v = int(2*y+t-8);
            if u < 0 or v < 0 or u > J.shape[0] or v > J.shape[1]:
                continue;
            
            J[u,v,:] = S[s,t]*Ix;

img.imsave("star_cuttlefish.jpg",J);
plt.imshow(J);