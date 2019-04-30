# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import numpy as np
import cv2
import time

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_images_labels(img_folder, label_file):
    with open(label_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        labels = []
        images = []
        for row in reader:
            img_filename = row[0]
            img = cv2.imread(os.path.join(img_folder,img_filename), 0)
            if img is not None:
                images.append(img.flatten()/255)
                labels.append(int(row[1]))
        return np.array(images), np.array(labels)
    
#transform data using PCA
def PCA(X, error):
    
    # calculate covariance matrix of transpose of X
    X_cov = np.cov(X.T)
    eig_vals, eig_vects = np.linalg.eig(X_cov)
    
    # rank eigenvectors and values from highest to lowest
    idx = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[idx]
    eig_vects = eig_vects[:,idx]
    
    # sum eigenvalues
    sigma_lamda = np.sum(eig_vals)
    
    basis_i = 0  
    
    # select axes with error rate < error
    for i in range(len(eig_vals)-1, -1, -1):
        
        vals_sum = np.sum(eig_vals[i:])
        
        if (vals_sum/sigma_lamda >= error):
            basis_i = i
            break
    
    # multiply dataset by optimal basis 
    P = np.dot(eig_vects[:, :basis_i+1].T , X.T)  
    
    return P.T

# function for printing confusion matrix and performance metrics
def get_metrics(y_pred, y_truth, print_metrics = False):
    tn = 0
    tp = 0 
    fn = 0 
    fp = 0
    
    for i in range(len(y_pred)):
        #calc TP
        if y_pred[i] == 1 and y_truth[i] == 1:
            tp += 1
    
        #calc TN 
        if y_pred[i] == 0 and y_truth[i] == 0:
            tn += 1
    
        #calc FP
        if y_pred[i] == 1 and y_truth[i] == 0:
            fp += 1
    
        #calc FN
        if y_pred[i] == 0 and y_truth[i] == 1:
            fn += 1
    
    if (print_metrics):
        # print confusion matrix
        print('\t\tPredicted Class')
        print('True Class\t', 'No', '\tYes')
        print('No\t\t', tn, '\t', fp)
        print('Yes\t\t', fn, '\t', tp )
        print()
    
    if (tn + tp + fn + fp) == 0.0:
        accuracy = 0.0
    else:
        accuracy = (tn + tp) / (tn + tp + fn + fp)
    
    if (tp + fn) == 0.0:
        sensitivity = 0.0
    else:
        sensitivity = tp / (tp + fn)  
    
    if (tp + fp) == 0.0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if (tn + fp) == 0.0:
        specificity = 0.0
    else:
        specificity = tn / (tn + fp)
    if print_metrics:
        print('Accuracy', accuracy)
        print('Sensitivity', sensitivity)
        print('Precision', precision)
        print('Specificity', specificity)
    
    return accuracy, sensitivity, precision, specificity, tp, tn, fp, fn

# function for finding MLE mean
def mle_mean(data):
    n = len(data)
    mean = np.sum(data, axis=0)/n
    return mean

# function for calculating covariance using MLE given mu
def mle_covariance(data, mu):
    
    n = len(data)
    diff = data - mu
    diff = diff.T
    product = np.matmul(diff, diff.T)
    covariance = product/n
    
    return covariance

# main maximum a-posteriori probability method
# returns test accuracy given test dataset and parameters
def mpp(X_train, y_train, X_test, prior_class_0, prior_class_1, case):
    
    # get dimension of data
    if X_train.ndim == 1:
        d = 1
        X_train = X_train.reshape(len(X_train), 1)
    else:
        d = X_train.shape[1]
        
    #print('d', d)
    
    # split training data into classes
    X_train_class_0 = X_train[y_train[:] == 0.0]
    X_train_class_1 = X_train[y_train[:] == 1.0]
    
    # predict class for each value in test dataset
    y_pred = []
    
    count = 0 
    for x in X_test:
        if count % 1000 == 0:
            print(count)
        count += 1
        
        x = x.reshape(d,1)
        
        # Case 1: Assume Gaussian PDF, covariance matrices are same, & independent features
        # Also known Euclidean Norm or Minimum Distance Classifier
        if case == 1:
            # get MLE mean for individual classes
            mean = mle_mean(X_train).reshape(d,1)
            mean_class_0 = mle_mean(X_train_class_0).reshape(d,1)
            mean_class_1 = mle_mean(X_train_class_1).reshape(d,1)
            # calculate single covariance matrix
            #print('calc cov')
            cov_mat = mle_covariance(X_train, mean.T)#np.cov(X_train[:,0], X_train[:,1])
            
            # calculate discriminant function for class 0
            v = x - mean_class_0
            g0 = -0.5*np.matmul(v.T, v)/cov_mat[0,0] + np.log(prior_class_0)

            # calculate discriminant function for class 1
            v = x - mean_class_1
            g1 = -0.5*np.matmul(v.T, v)/cov_mat[0,0] + np.log(prior_class_1)
            
        # Case 2: Assume Gaussian PDF & covariance matrices are same
        # Also known Squared Mahalanobis Distance Classifier
        elif case == 2:
            # get MLE mean for individual classes
            mean = mle_mean(X_train).reshape(d,1)
            mean_class_0 = mle_mean(X_train_class_0).reshape(d,1)
            mean_class_1 = mle_mean(X_train_class_1).reshape(d,1)
            
            # calculate single covariance matrix
            cov_mat = mle_covariance(X_train, mean.T)#cov_mat = np.cov(X_train[:,0], X_train[:,1])
            invC = np.linalg.inv(cov_mat)
            
            # calculate discriminant function for class 0
            v = x - mean_class_0
            g0 = -0.5*np.matmul(np.matmul(v.T, invC), v) + np.log(prior_class_0)

            # calculate discriminant function for class 1
            v = x - mean_class_1
            g1 = -0.5*np.matmul(np.matmul(v.T, invC), v) + np.log(prior_class_1)

        # Case 3: Assume Gaussian PDF only
        # Equivalent of MPP method
        else:
            
            # get MLE mean for individual classes
            mean = mle_mean(X_train).reshape(d,1)
            mean_class_0 = mle_mean(X_train_class_0).reshape(d,1)
            mean_class_1 = mle_mean(X_train_class_1).reshape(d,1)
            
            # calculate covariance matrices for both classes
            #print(X_train_class_0, mean_class_0.T)
            cov_mat0 = mle_covariance(X_train_class_0, mean_class_0.T)
            cov_mat1 = mle_covariance(X_train_class_1, mean_class_1.T)
            
            # calculate discriminant function for class 0
            v = x - mean_class_0
            invC = np.linalg.inv(cov_mat0)
            #print(np.linalg.det(cov_mat0))
            g0 = -0.5*np.matmul(np.matmul(v.T, invC), v) - 0.5*np.log(np.linalg.det(cov_mat0)) + np.log(prior_class_0)
            # calculate discriminant function for class 1
            v = x - mean_class_1
            invC = np.linalg.inv(cov_mat1)
            g1 = -0.5*np.matmul(np.matmul(v.T, invC), v) - 0.5*np.log(np.linalg.det(cov_mat1)) + np.log(prior_class_1)
        
        # append classification 
        if g0 > g1:
            y_pred.append(0.0)
        else:
            y_pred.append(1.0)
            
    return y_pred

# function for training MPP classifier with 5-fold cross validation
def train_mpp_with_validation(X_train, y_train, prior_class_0, prior_class_1, case, folds=5):  
    
    accuracies = []
    models = []
    
    # k fold cross validation
    for f in range(0, folds):
        print('fold', f)

        n = len(X_train)
        split_ind_start = int (f * n / folds) 
        split_ind_stop  = int ((f+1) * n / folds)
            

        if f == 0:
            X_train_tmp = X_train[int(split_ind_stop):]
            y_train_tmp = y_train[int(split_ind_stop):]

        elif f == folds-1:
            X_train_tmp = X_train[:int(split_ind_start)]
            y_train_tmp = y_train[:int(split_ind_start)]

        else:
            if X_train.ndim != 1:
                X_train_tmp = np.concatenate([X_train[:int(split_ind_start)], X_train[int(split_ind_stop):]], axis = 0)
                y_train_tmp = np.concatenate([y_train[:int(split_ind_start)], y_train[int(split_ind_stop):]], axis = 0)
            else:
                X_train_tmp = np.concatenate((X_train[:int(split_ind_start)], X_train[int(split_ind_stop):]))
                y_train_tmp = np.concatenate((y_train[:int(split_ind_start)], y_train[int(split_ind_stop):]))


        X_valid = X_train[int(split_ind_start):int(split_ind_stop)]
        y_valid = y_train[int(split_ind_start):int(split_ind_stop)]
        
        # MPP Case 1
        if case == 1:
            y_pred_valid = mpp(X_train, y_train, X_valid, prior_class_0, prior_class_1, case)
        
        # MPP Case 2
        elif case == 2:
            y_pred_valid = mpp(X_train, y_train, X_valid, prior_class_0, prior_class_1, case)
        
        # MPP Case 3
        else:
            y_pred_valid = mpp(X_train, y_train, X_valid, prior_class_0, prior_class_1, case)
            
        performance_metrics = get_metrics(y_valid, y_pred_valid)
        accuracy = performance_metrics[0]
        print('Fold', f + 1, 'accuracy:', accuracy)
        accuracies.append(accuracy)
        models.append(performance_metrics)
        
    print('Mean validation accuracy:', np.mean(accuracies))
    
    mean_performance_metrics = np.mean(np.array(models), axis = 0)
    
    return mean_performance_metrics

# function for calculating Euclidean distance
def euc_dist(x, y):
    
    diff = x-y
    
    if(diff.ndim == 1):
        return np.abs(diff)
    
    return np.linalg.norm(diff, axis = 1)

# k-nearest neighbors implementation
def knn(X, y, X_te, prior_class_0, prior_class_1, k):
    
    start_time = time.time()
    
    predictions = np.zeros(len(X_te))
    
    indices = np.array([ind for ind in range(len(X))])
    
    prior_class_0_actual = len(X[y[:] == 0.0])/len(X)
    prior_class_1_actual = len(X[y[:] == 1.0])/len(X)
    
    for i in range(len(X_te)):
        s = euc_dist(X_te[i], X)
        s = np.column_stack((indices, s))
        s = s[s[:,1].argsort()]
        
        # get k nearest neighbors
        knns = s[:k]
        classes = []
        
        for ind in knns[:,0]:
            classes.append(int(y[int(ind)]))
    
        k_0 = classes.count(0)
        
        n = len(X)
        
        r_0 = prior_class_0 * prior_class_0_actual
        
        g_0 = k_0 / k * r_0
        
        k_1 = classes.count(1)
    
        g_1 = k_1 / k * (1.0 - prior_class_0_actual * r_0)
    
        predictions[i] = g_0 < g_1
    
    end_time = time.time()
    
    print("Elapsed Time:", end_time - start_time)
    
    return predictions

def train_knn_with_validation(X_train, y_train, prior_class_0, prior_class_1, k, folds=5):  
    
    accuracies = []
    models = []
    
    # k fold cross validation
    for f in range(0, folds):

        n = len(X_train)
        split_ind_start = int (f * n / folds) 
        split_ind_stop  = int ((f+1) * n / folds)
            

        if f == 0:
            X_train_tmp = X_train[int(split_ind_stop):]
            y_train_tmp = y_train[int(split_ind_stop):]

        elif f == folds-1:
            X_train_tmp = X_train[:int(split_ind_start)]
            y_train_tmp = y_train[:int(split_ind_start)]

        else:
            if X_train.ndim != 1:
                X_train_tmp = np.concatenate([X_train[:int(split_ind_start)], X_train[int(split_ind_stop):]], axis = 0)
                y_train_tmp = np.concatenate([y_train[:int(split_ind_start)], y_train[int(split_ind_stop):]], axis = 0)
            else:
                X_train_tmp = np.concatenate((X_train[:int(split_ind_start)], X_train[int(split_ind_stop):]))
                y_train_tmp = np.concatenate((y_train[:int(split_ind_start)], y_train[int(split_ind_stop):]))

        X_valid = X_train[int(split_ind_start):int(split_ind_stop)]
        y_valid = y_train[int(split_ind_start):int(split_ind_stop)]
        
        # KNN
        y_pred_valid = knn(X_train_tmp, y_train_tmp, X_valid, prior_class_0, prior_class_1, k)
        
        performance_metrics = get_metrics(y_valid, y_pred_valid)
        accuracy = performance_metrics[0]
        print('Fold', f + 1, 'accuracy:', accuracy)
        accuracies.append(accuracy)
        models.append(performance_metrics)
    
    print('Mean validation accuracy:', np.mean(accuracies))
    
    mean_performance_metrics = np.mean(np.array(models), axis = 0)
    
    return mean_performance_metrics