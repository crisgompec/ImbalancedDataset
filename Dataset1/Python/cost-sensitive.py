#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost-Sensitive Classification
"""

import numpy as np
from scipy.stats import multivariate_normal as Normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import GaussianNB


my_colors = [(0,0,1),(1,0,0)]
my_cm = LinearSegmentedColormap.from_list('my_cm', my_colors, N=2)

    
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def GaussianBayesClassifier(x,priors, gaussians):
    pyx1 = multivariate_gaussian(x, gaussians[0][0], gaussians[0][1])
    pyx2 = multivariate_gaussian(x, gaussians[1][0], gaussians[1][1])
    
    y_pred = np.zeros(x.shape[:2])
    for i in range(len(x)):
        for j in range(len(x)):
            #print('%.3f vs. %.3f' % (np.log(priors[0]) + np.log(pyx1[i,j]), np.log(priors[1]) + np.log(pyx2[i,j])))
            if np.log(priors[0]) + np.log(pyx1[i,j]) > np.log(priors[1]) + np.log(pyx2[i,j]):
                y_pred[i,j] = -1
            else:
                y_pred[i,j] = 1
    return y_pred

def hBayes(x, priors, gaussians):
    h0 = priors[0] * Normal.pdf(x,gaussians[0][0], gaussians[0][1])
    h1 = priors[1] * Normal.pdf(x, gaussians[1][0], gaussians[1][1])
    y_pred = np.zeros([len(h0),1])
    for i in range(len(h0)):
        if h0[i] > h1[i]:
            y_pred[i] = -1
        else:
            y_pred[i] = 1
    return y_pred

def Plot2DGaussians(grid, Z1, Z2, Zpred):
    x_grid, y_grid = grid
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1,projection='3d')
    
    ax1.plot_surface(x_grid, y_grid, Z2, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis, alpha=0.8)
    ax1.plot_surface(x_grid, y_grid, Z1, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.Spectral, alpha=0.7)
    
    ax1.view_init(20,-70)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    
    ax2 = fig.add_subplot(3,1,2,projection='3d')
    ax2.contourf(x_grid, y_grid, Z2, zdir='z', offset=0, cmap=cm.viridis, alpha=0.8, antialiased=True)
    ax2.contourf(x_grid, y_grid, Z1, zdir='z', offset=0, cmap=cm.Spectral, alpha=0.5, antialiased=True)
    ax2.view_init(90, 270)
    
    ax2.grid(False)
    ax2.set_zticks([])
    ax2.set_xlabel(r'Coordinate $x_1$')
    ax2.set_ylabel(r'Coordinate $x_2$')
    
    ax3 = fig.add_subplot(3,1,3)
    plt.contourf(x_grid,y_grid,Zpred,levels=[-1,0,1], cmap = my_cm, alpha=0.2)
    plt.contour(x_grid,y_grid,Zpred,linewidth=2,colors='k')
    ax3.set_xlabel(r'Coordinate $x_1$')
    ax3.set_ylabel(r'Coordinate $x_2$')
    ax3.set_title(r'Decision Region')
    
    plt.show()
    return

def PlotDataAndClassification(grid, X1, X2, Zpred, title):
    x_grid,y_grid = grid
    plt.figure()
    
    # Plot data points
    plt.plot(X1[:,0],X1[:,1],'o',color='b',label='Class 1')
    plt.plot(X2[:,0],X2[:,1],'o',color='r',label='Class 2')
    
    # Plot classification region
    plt.contourf(x_grid,y_grid,Zpred,levels=[-1,0,1], cmap = my_cm, alpha=0.2)
    plt.contour(x_grid,y_grid,Zpred,linewidth=2,colors='k')
    
    # Set labels
    plt.xlabel(r'Coordinate $x_1$')
    plt.ylabel(r'Coordinate $x_2$')
    plt.legend(loc='lower right',fontsize='x-small')
    plt.title(title)

    return

# Binary NB Classifier
class NaiveBayes:
    def __init__(self, x=None,y=None,c=None):
        try:
            self.fit(x,y,c)
        except:
            pass
    
    def fit(self, x, y, c = None):
        x1, x2 = x[np.where(y>0.5)[0]], x[np.where(y<0.5)[0]]
        y1, y2 = y[np.where(y>0.5)[0]], y[np.where(y<0.5)[0]]
        
        self.cov1, self.cov2 = np.zeros([2,2]), np.zeros([2,2])
        
        self.mean1 = np.mean(x1,0)
        self.mean2 = np.mean(x2,0)
        print(self.mean2)
        self.cov1[0,0], self.cov1[1,1] = np.var(x1,0)[0], np.var(x1,0)[1]
        self.cov2[0,0], self.cov2[1,1] = np.var(x2,0)[0], np.var(x2,0)[1]
        
        self.gaussians = [[self.mean1,self.cov1],[self.mean2, self.cov2]]
        print(self.gaussians)
        self.prior1 = len(y1)/(len(y1)+len(y2))
        self.prior2 = len(y2)/(len(y1)+len(y2))
        if c is not None:
            self.prior1 *= c[0]
            self.prior2 *= c[1]
    
    def predict(self, x):
        return hBayes(x,[self.prior1, self.prior2], self.gaussians)
        
# TODO: functions to evaluate risk and perfomance data and classifier (we will use it later to try different costs)



if __name__ == '__main__':
    x_grid = np.linspace(-8,8,100)
    y_grid = np.linspace(-8,8,100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(x_grid.shape + (2,))
    pos[:, :, 0] = x_grid
    pos[:, :, 1] = y_grid
    
    # Part 1: Definition of the Problem and Motivation
    # Define Gaussians
    mean1 = [0,1]
    mean2 = [-1,-0.5]
    cov1 = [[4,0],[0,2]]
    cov2 = [[1,0],[0,1]]
    gaussians = [[np.array(mean1),np.array(cov1)],[np.array(mean2), np.array(cov2)]]
    
    Z1 = multivariate_gaussian(pos, np.array(mean1), np.array(cov1))
    Z2 = multivariate_gaussian(pos, np.array(mean2), np.array(cov2))
    Z_bayes_balanced = GaussianBayesClassifier(pos,[0.5, 0.5], gaussians)
    
    # Plot Distributions
    Plot2DGaussians([x_grid,y_grid], Z1, Z2, Z_bayes_balanced)
    
    # Make assumptions on priors and plot (one can force trivial classification setting prior1=0.99)
    prior1 = 0.85
    prior2 = 1-prior1
    Z_bayes_unbalanced = GaussianBayesClassifier(pos,[prior1, prior2], gaussians)
    Plot2DGaussians([x_grid,y_grid], Z1*prior1, Z2*prior2, Z_bayes_unbalanced)

    # RESULTS
    # 1) Create data
    N = 1000
    X1 = np.random.multivariate_normal(mean1,cov1,int(round(N*prior1)))
    X2 = np.random.multivariate_normal(mean2,cov2,int(round(N*prior2)))
    
    # 2) Plot data and classification
    # gnb = GaussianNB().fit(np.r_[X1,X2],np.r_[-np.ones([int(round(N*prior1)),1]),np.ones([int(round(N*prior2)),1])])
    # Ypred = hBayes(np.r_[X1,X2],[prior1, prior2], gaussians)
    # Ypred = gnb.predict(np.r_[X1,X2])
    # Create NB non cost-sensitive
    NB_Classifier = NaiveBayes()
    NB_Classifier.fit(np.r_[X1,X2],np.r_[-np.ones([int(round(N*prior1)),1]),np.ones([int(round(N*prior2)),1])])
    Ypred = NB_Classifier.predict(np.r_[X1,X2])
    
    # 3) Plot classification
    PlotDataAndClassification([x_grid,y_grid], X1, X2, Z_bayes_unbalanced, 'Results without cost-sensitive classification')
    
    # 4) Evaluate Risk
    
    
    # 5) Evaluate Perfomance
    # Metrics: pcf, F1, ROC, 
    
    
    # %% Part 2: Applying Thresholded Maximum Likelihood
    
    
    # %% Part 3: Applying Weighted Likelihood
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    