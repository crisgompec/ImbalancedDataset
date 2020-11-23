#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

"""
Classifiers Library
This library includes 5 different classifiers:
    Bagging
    Adaboost
    AdaCost
    BoostingSVM
    AdaMEC
"""


""" Custom Bagging implementation"""
class Bagging_classifier:
    def __init__(self, type_classifier, x=None, y=None, n_iterations = 20, ratio=0.1):
        
        """
        type_classifier: DecisionTree, KNN, SVM, GaussianNB
        """
        self.number_iterations = n_iterations
        self.type_classifier = type_classifier 
        
        self.ratio = ratio #Ratio Bootstrapped dataset/ original dataset
        
                
    def fit(self,X_train,y_train):
        
        dataset_train = np.concatenate((X_train.to_numpy(), y_train),axis = 1)
        N = np.shape(dataset_train)[0]
        
        # There will be as many classifier models as iterations
        self.classifier_models = np.zeros(shape=self.number_iterations, dtype=object)
        
        for classifier_iteration in range(self.number_iterations):
    
            dataset_train_undersampled = dataset_train[random.sample(range(1,N),int(self.ratio*N)), :]

            X_train_undersampled = dataset_train_undersampled[:,0:59]
            y_train_undersampled = dataset_train_undersampled[:,59].astype(int)


            ### Train different algorithms

            # Decision tree
            if self.type_classifier == "DecisionTree": 
                classifier = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
                classifier_model = classifier.fit(X_train_undersampled, y_train_undersampled)


            # K-NN
            elif self.type_classifier == "KNN":
                classifier = KNeighborsClassifier(n_neighbors=3)
                classifier_model = classifier.fit(X_train_undersampled, y_train_undersampled)

        
            # SVM
            elif self.type_classifier == "SVM":
                classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                classifier_model = classifier.fit(X_train_undersampled, y_train_undersampled)

            # Gaussian Naive Bayes    
            elif self.type_classifier == "GaussianNB":
                classifier = GaussianNB()
                classifier_model = classifier.fit(X_train_undersampled, y_train_undersampled) 
            else:
                print("Wrong classifier selection")
                return
                
            self.classifier_models[classifier_iteration] = classifier_model
        
        return
    
    
    def predict(self,X_test):
        
        model_preds = np.array([model.predict(X_test) for model in self.classifier_models])
        y_test_pred = np.sign(np.mean(model_preds,axis = 0))
        return y_test_pred.astype(int)
            

""" Custom AdaBoost Classifier implementation"""
class AdaboostClassifier:
    def __init__(self, x=None,y=None, n_iterations = 100):
        self.number_iterations = n_iterations
    
    def fit(self,X_train,y_train):
        number_samples = np.shape(X_train)[0]
        weights = np.ones(number_samples)/number_samples
        
        # There will be as many weak learners as iterations
        self.weak_learners = np.zeros(shape=self.number_iterations, dtype=object)
        self.significance_vec = np.zeros(shape=self.number_iterations)
        #error_vec = []
        #accuracy_vec = []
        
        for iterations in range(self.number_iterations):
            current_weights = weights

            weak_learner = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            weak_learner_model = weak_learner.fit(X_train, y_train, sample_weight=current_weights)

            # The new weak learner model is saved
            self.weak_learners[iterations] = weak_learner_model
            weak_learner_pred = weak_learner_model.predict(X_train)

            error = 0
            incorrect_pred = 0
            correct_pred = 0
            for item_index in range(number_samples):
                if weak_learner_pred[item_index] != y_train[item_index]:
                    incorrect_pred = incorrect_pred + 1
                    error = error + current_weights[item_index]
                else: 
                    correct_pred = correct_pred + 1 

            # Save error for plotting    
            # error_vec.append(error)

            # Significance of the weak learner model is calculated and saved
            significance = 0.5*np.log((1-error)/error) 
            self.significance_vec[iterations] = significance

            # Update weights for each sample
            for item_index in range(number_samples):
                if weak_learner_pred[item_index] != y_train[item_index]:
                    weights[item_index] = np.multiply(current_weights[item_index],np.exp(significance))
                else:
                    weights[item_index] = current_weights[item_index]*np.exp(-significance)

            # Normalize weights
            weights /= weights.sum()
        
    def predict(self,X_test):
        model_preds = np.array([model.predict(X_test) for model in self.weak_learners])
        y_test_pred = np.sign(np.dot(self.significance_vec, model_preds))
        return y_test_pred.astype(int)

""" Custom AdaCost Classifier implementation"""
class AdaCost:
    """This is the implementation of the AdaCost Classifier. In this algorithm, 
    the weight update is modified by adding a cost adjustment function φ. 
    This function, for an instance with a higher cost factor increases its weight 
    “more” if the instance is misclassified, but decreases its weight “less” otherwise.
    
    This implementation uses the function φ = +/-0.5*cost + 0.5
    
    Requires:
        X_train: Training features. Size N x D
        y_train: Training labels. Size N x 1
        cost: cost used to update weight via te adjustment function

    """
    
    def __init__(self, x=None,y=None, cost = None, n_iterations = 100):
        self.number_iterations = n_iterations
        if x != None and y != None and cost != None:
            self.fit(x,y,cost)
        
    def fit(self,X_train,y_train, cost):
        # Initialize weights
        number_samples = len(X_train)
        weights = np.ones(number_samples)/number_samples
        
        # Define adjustment function φ (called beta)
        beta = 0.5*np.ones(number_samples)
        beta[np.where(y_train==1)[0]] += cost*0.5
        beta[np.where(y_train==-1)[0]] -= cost*0.5
        
        # Define vectors to store weak predictors and significances of each iteration
        self.weak_learners = np.zeros(shape=self.number_iterations, dtype=object)
        self.significance_vec = np.zeros(shape=self.number_iterations)
        
        for it in range(self.number_iterations):
            current_weights = weights
            
            # Create and save week learner for this iteration
            weak_learner_model = DecisionTreeClassifier(max_depth=1, 
                                 max_leaf_nodes=2).fit(X_train, y_train, sample_weight=current_weights)
            self.weak_learners[it] = weak_learner_model
            weak_learner_pred = weak_learner_model.predict(X_train)
            
            # Calculate r
            u = np.multiply(np.multiply(weak_learner_pred, y_train),beta)
            r = np.sum(np.multiply(weights,u))
            
            # Calculate and store significance of this iteration
            significance = 0.5 * np.log((1+r)/(1-r))
            self.significance_vec[it] = significance
            
            # Update weights for next iteration
            weights = np.multiply(weights,np.exp(-significance * u))
            weights /= weights.sum()    
            
            # Round Debugging
            #print('Round %d' % (it))
            #print(u)
            #print(r)
            #print(significance)
            #print(weights)
            
            #input()

        
    def predict(self,X_test):
        model_preds = np.array([model.predict(X_test) for model in self.weak_learners])
        y_test_pred = np.sign(np.dot(self.significance_vec, model_preds))
        return y_test_pred.astype(int)

""" Custom BoostingSVM Classifier implementation"""
class BoostingSVM:
    def __init__(self, x=None,y=None, sigma_ini = 50, sigma_min = 10, sigma_step = 0.5):
        """
        This is a custom implementation of the Boosting SVM algorithm. It is based 
        on the combination of Adaboost and RFB SVM weak learners. To create the model,
        this class needs the following inputs:
        
        X_train: Training features. Size N x D
        y_train: Training labels. Size N x 1

        """
        self.sigma_ini = sigma_ini
        self.sigma_step = sigma_step
        self.sigma_min = sigma_min
    
    def fit(self,X_train,y_train):
        sigma = self.sigma_ini 
        
        # Initialize weights
        number_samples = np.shape(X_train)[0]
        weights = np.ones(number_samples)/number_samples
        
        # Define vectors to store weak predictors and significances of each iteration
        self.weak_learners = [] #np.zeros(shape=self.number_iterations, dtype=object)
        self.significance_vec = [] #np.zeros(shape=self.number_iterations)
        self.error_debug = []
        
        # Todo: Apply dimensionality reduction
        self.pca = PCA(n_components = 2).fit(X_train)
        self.variance_explained = self.pca.explained_variance_ratio_
        X_train = self.pca.fit_transform(X_train)
        
        #for iterations in range(self.number_iterations):
        while sigma > self.sigma_min:
            print('Sigma: %.1f' % sigma)
            #print('BoostSVM iteration: %d' % (iterations))
            current_weights = weights
            
            # Create and save week learner for this iteration
            weak_learner = SVC(kernel='rbf', gamma = 1/2/sigma**2) #SVC(max_iter=10,tol=5)
            weak_learner_model = weak_learner.fit(X_train, y_train, sample_weight=current_weights)

            # The new weak learner model is saved
            weak_learner_pred = weak_learner_model.predict(X_train)
            
            # Calculate error
            error = np.sum(current_weights[np.where(weak_learner_pred != y_train)[0]]) 
            self.error_debug.append(error)
            
            if error > 0.5:
                sigma = sigma - self.sigma_step
            else:
                # Significance of the weak learner model is calculated and saved
                significance = 0.5*np.log((1-error)/error) 
                self.significance_vec.append(significance)
                self.weak_learners.append(weak_learner_model)

                # Update weights for each sample
                idx_incorrect = np.where(weak_learner_pred != y_train)[0]
                idx_correct = np.where(weak_learner_pred == y_train)[0]
                weights[idx_incorrect] = np.multiply(current_weights[idx_incorrect],np.exp(significance))
                weights[idx_correct] = current_weights[idx_correct]*np.exp(-significance)

                # Normalize weights
                weights /= weights.sum()
        
    def predict(self,X_test):
        X_test_pca = self.pca.fit_transform(X_test)
        model_preds = np.array([model.predict(X_test_pca) for model in self.weak_learners])
        y_test_pred = np.sign(np.dot(self.significance_vec, model_preds))
        return y_test_pred.astype(int)

""" AdaMEC Classifier implementation
Based on:
Nikolaou, N., Edakunni, N. U., Kull, M., Flach, P. A., and Brown, G., 
'Cost-sensitive boosting algorithms: Do we really need them?', 
Machine Learning, 104(2), pages 359-384, 2016. 
[http://link.springer.com/article/10.1007/s10994-016-5572-x]
"""
class AdaMEC:
    def __init__(self, x=None,y=None, n_iterations = 100):
        self.number_iterations = n_iterations
    
    def fit(self, X_train, y_train):
        X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.5)
        
        #Train an AdaBoost ensemble
        AdaBoostUncal = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators= self.number_iterations)
        AdaBoostUncal = AdaBoostUncal.fit(X_train, y_train)
	
        #Now calibrate the ensemble on the data reserved for calibration
        self.AdaBoostCal = CalibratedClassifierCV(AdaBoostUncal, cv="prefit", method='sigmoid')
        self.AdaBoostCal.fit(X_cal, y_cal)

        return self.AdaBoostCal

    def predict(self, threshold, X_test):
        scores_CalibratedAdaMEC = self.AdaBoostCal.predict(X_test)[:,1] #Positive Class scores
        y_pred_CalibratedAdaMEC = -np.ones(X_test.shape[0])
        y_pred_CalibratedAdaMEC[np.where(scores_CalibratedAdaMEC > threshold)] = 1#Classifications, AdaMEC uses a shifted decision threshold (skew-sensitive) 

        return y_pred_CalibratedAdaMEC

