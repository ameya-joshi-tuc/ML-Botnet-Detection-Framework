#!/usr/bin/python

#######################################################################
#---------------------------------------------------------------------#
#Import Libraries
#---------------------------------------------------------------------#

import random
import time			#To add delay
import subprocess	#To invoke bash commands and background scripts
import os
import csv
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
sns.set(style="ticks")

import numpy as np
import pandas as pd

np.set_printoptions(precision=None, suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 7)
pd.set_option('display.width', 1000)

from tabulate import tabulate
import datetime
import itertools

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.neural_network import BernoulliRBM
from joblib import dump, load

#Fix Numpy Seed
np.random.seed(57)

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
#Global Variables
#---------------------------------------------------------------------#

NORMAL_TRAIN_DATA = 'normal_train.csv'			#Normal training data
ANOMALOUS_DATA = 'anomalous_validation.csv'		#Anomalous validation/testing data
NORMAL_TEST_DATA = 'normal_validation.csv'		#Normal validation/testing data

ENSEMBLE_TRAIN_DATA = 'Z_val_OLAK_pred.csv'		#Concatenated Z matrix for ensemble obtained from validation data
ENSEMBLE_TEST_DATA = 'Z_test_OLAK_pred.csv'		#Concatenated Z matrix for ensemble obtained from test data		

#Ensemble Mode
SML_ENSEMBLE = True			#To use SML
RBM_ENSEMBLE = False		#To use RBM
ENSEMBLE_TRAIN = False		#To train the RBM ensemble/Use validation data

#Statistical Data Analysis
REAL_DATA_ANALYSIS = True		#True for real data analysis, False for artificial data analysis

#RBM Ensemble
(N_COMPONENTS, LEARNING_RATE, BATCH_SIZE, N_ITERATIONS) = (1, 0.002, 10, 100)
#Ensemble Threshold Percentages; Anomalous<=T1, T1<Normal (Give in decimal form, e.g. 0.7)
T1 = 0.5

##### While using artifical data #####
#Generating data as per current dataset
TL = np.ones((105384,))
TL[52692:]=-1

#For Z_new_generator:	[TPR, FPR]
classifier_stats_val = np.array(([0.8642, 0.1187], [0.7513, 0.0936], [0.9549, 0.09974], [0.83604, 0.1117]))
classifier_stats_test = np.array(([0.8656, 0.1215], [0.756, 0.0892], [0.9599, 0.09943], [0.8384, 0.1126]))

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def Z_new_generator(data_length, classifier_stats):
	'''
	Generate artificial data using TPR and FPR values obtained from real dataset
	
	Returns: Array of True Labels and the Z matrix which contains artificially generated predictions
	'''
	num_classifiers = classifier_stats.shape[0]
	half_length = int(data_length/2)
	#Generate True Labels
	TL = np.ones((data_length,))
	TL[half_length:]=-1
	#Generate Z
	Z = np.zeros((data_length, num_classifiers))
	for n in range(num_classifiers):
		print(f'\nClassifier under consideration - {n}')
		tpr = classifier_stats[n,0]
		fpr = classifier_stats[n,1]
		Z[0:half_length,n] = np.random.choice([-1, 1], size= (half_length,), p = [fpr, 1-fpr])		#Normal data
		Z[half_length:,n] = np.random.choice([-1, 1], size= (half_length,), p = [tpr, 1-tpr])		#Anomalous Data

		#Z check
		true_positives = np.count_nonzero(Z[half_length:,n]==-1)
		false_positives = np.count_nonzero(Z[0:half_length,n]==-1)
		false_negatives = np.count_nonzero(Z[half_length:,n]==1)
		true_negatives = np.count_nonzero(Z[0:half_length,n]==1)
		
		print(tabulate([['Predicted Anomalous', true_positives, false_positives], \
						['Predicted Normal', false_negatives, true_negatives]], \
				headers=[' ', 'Actual Anomalous', 'Actual Normal'], tablefmt='orgtbl'))
	
	return (TL, Z)
	
#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def classifier_rank_estimator(Z):
	'''
	Estimate the ranks of the classifier by computing the leading Eigen Vector of the Rank-1 Matrix
	
	Returns : 	Leading Eigen-vector of size 1 by m where each column represents weight of corresponding classifier
				e.g. for 4 classifiers, returns [v1 v2 v3 v4] where v(i) indicates the weight of the ith classifier
				(m - Number of predictors)
	'''
	Q = np.cov(Z.transpose())
	#print('\nCovariance Matrix - \n {} {}'.format(Q, Q.shape))
	R = Q.copy()
	#https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
	m = Q.shape[0]			
	r = np.arange(m)
	mask = r[:,None] < r
	upper_diag = Q[mask]
	#print('\nUpper diagonal entries - \n', upper_diag)
	B = np.array(np.log(abs(upper_diag)))
	#print('\nVector B - \n', B)
	A = np.zeros((B.shape[0],m))
	count = 1
	temp = 0
	for j in range(m):
		for k in range(count,m):
			A[temp,j] = 1
			A[temp,k] = 1
			temp = temp + 1
		count = count + 1
	#print('\nMatrix A - \n',A)
	X, residuals, rank, S = np.linalg.lstsq(A,B,rcond=None)
	#print('\nSolution X - \n', X)
	#print('\nQ matrix is - \n', R)
	for i in range(m):
		R[i,i] = np.exp(2*X[i])
	#print('\nR matrix is - \n', R)
	eig_val, eig_vect = np.linalg.eig(R)
	#print('\nEigen Values - \n', eig_val)
	#print('\nEigen Vectors - \n', eig_vect)
	return eig_vect[:, np.argmax(eig_val)]

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def SML(v, Z):
	'''
	Obtain final predictions based on Spectral Meta-Learner
	Returns : Scalar final prediction of 1 or -1 as decided by the Ensemble
	'''
	y_SML = np.sign(np.sum(abs(v)*Z, axis=1))
	return y_SML

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def ml_analysis():
	'''
	Evaluate the Ensemble performances from the prediction data
	'''
	def RBM_ensemble(predict_all):
		'''
		RBM Ensemble method
		'''
		#Convert '-1's to '0'
		predict_all[predict_all==-1]=0
		
		if ENSEMBLE_TRAIN==True:
			#Train RBM 
			rbm = BernoulliRBM(n_components = N_COMPONENTS, n_iter=N_ITERATIONS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
			print(f'\nStarting RBM training.... {datetime.datetime.now().time()}')
			rbm.fit(predict_all)
			print(f'\nRBM training complete.... - {datetime.datetime.now().time()}')
			#dump(rbm, 'rbm_ensemble_OLAK.joblib', compress=True)	#Save RBM model
			anomalous_test_predict_all = predict_all[5855:,:]
			normal_test_predict_all = predict_all[0:5855,:]
		elif ENSEMBLE_TRAIN==False:
			rbm = load('rbm_ensemble_OLAK.joblib')					#Load stored RBM model
			anomalous_test_predict_all = predict_all[52692:,:]
			normal_test_predict_all = predict_all[0:52692,:]
			print(f'Sizes - {anomalous_test_predict_all.shape}')
		
		anomalous_test_probability = rbm.transform(anomalous_test_predict_all)
		normal_test_probability = rbm.transform(normal_test_predict_all)
		print(f'\nRBM complete.... - {datetime.datetime.now().time()}')
		
		true_positives = int((anomalous_test_probability[anomalous_test_probability<=T1].shape[0])/N_COMPONENTS)
		false_negatives = int((anomalous_test_probability[T1<anomalous_test_probability].shape[0])/N_COMPONENTS)
		false_positives = int((normal_test_probability[normal_test_probability<=T1].shape[0])/N_COMPONENTS)
		true_negatives = int((normal_test_probability[T1<normal_test_probability].shape[0])/N_COMPONENTS)
		MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
	
	########################################################################################
	#ML_ANALYSIS code
	print('******** STARTING ENSEMBLE EVALUATIONS ********')
	if RBM_ENSEMBLE==True:
		if ENSEMBLE_TRAIN == True:
			predict_all = np.loadtxt(ENSEMBLE_TRAIN_DATA, delimiter=',')
		elif ENSEMBLE_TRAIN == False:
			predict_all = np.loadtxt(ENSEMBLE_TEST_DATA, delimiter=',')
		print('\nRBM predictions on data....\n')
		RBM_ensemble(predict_all)
	
	if SML_ENSEMBLE==True:
		train_predict_all = np.loadtxt(ENSEMBLE_TRAIN_DATA, delimiter=',')
		test_predict_all = np.loadtxt(ENSEMBLE_TEST_DATA, delimiter=',')
		
		print(f'Evaluating data size - {test_predict_all.shape}')
		
		train_half_length = int(train_predict_all.shape[0]/2)
		test_half_length = int(test_predict_all.shape[0]/2)
		
		#Calculate leading Eigenvector from Validation Data
		v = classifier_rank_estimator(train_predict_all)
		
		#SML on validation data
		print(f'\n\n****VALIDATION SML RESULTS*****')
		y = SML(v, train_predict_all)
		true_positives = sum(y[train_half_length:]==-1)
		false_negatives = sum(y[train_half_length:]==1)
		false_positives = sum(y[0:train_half_length]==-1)
		true_negatives = sum(y[0:train_half_length]==1)
		MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
		
		#SML on test data
		print(f'\n\n****TEST SML RESULTS*****')
		y = SML(v, test_predict_all)
		true_positives = sum(y[test_half_length:]==-1)
		false_negatives = sum(y[test_half_length:]==1)
		false_positives = sum(y[0:test_half_length]==-1)
		true_negatives = sum(y[0:test_half_length]==1)
		MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
	
#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def RBM_new():
	'''
	RBM to evaluate on artificially generated data
	'''
	#Define datasize
	(val_length, test_length) = (11710, 105384)
	(train_half_length, test_half_length) = (int(val_length/2), int(test_length/2))
	
	#Generate artificial Z
	print('\n##############VALIDATION DATA#################')
	(TL_val, Z_val) = Z_new_generator(val_length, classifier_stats_val)
	print('\n##############TEST DATA#################')
	(TL_test, Z_test) = Z_new_generator(test_length, classifier_stats_test)
	
	#Convert '-1's to '0'
	Z_val[Z_val==-1]=0
	Z_test[Z_test==-1]=0
	
	#Train RBM
	rbm = BernoulliRBM(n_components = N_COMPONENTS, n_iter=N_ITERATIONS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
	print(f'\nStarting RBM training.... {datetime.datetime.now().time()}')
	Z_val_probability = rbm.fit_transform(Z_val)
	Z_test_probability = rbm.transform(Z_test)
	print(f'\nRBM complete.... - {datetime.datetime.now().time()}')
	
	#Convert probability to values
	Z_val_final = np.sign(Z_val_probability - T1)
	Z_test_final = np.sign(Z_test_probability - T1)
	
	#RBM on validation data
	print(f'\n\n****VALIDATION RBM RESULTS*****')
	true_positives = sum(Z_val_final[train_half_length:]==-1)
	false_negatives = sum(Z_val_final[train_half_length:]==1)
	false_positives = sum(Z_val_final[0:train_half_length]==-1)
	true_negatives = sum(Z_val_final[0:train_half_length]==1)
	MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
	
	#RBM on test data
	print(f'\n\n****TEST RBM RESULTS*****')
	true_positives = sum(Z_test_final[test_half_length:]==-1)
	false_negatives = sum(Z_test_final[test_half_length:]==1)
	false_positives = sum(Z_test_final[0:test_half_length]==-1)
	true_negatives = sum(Z_test_final[0:test_half_length]==1)
	MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
	
#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def SML_new():
	'''
	SML on new generated Z matrix
	'''
	
	#Define datasize
	(val_length, test_length) = (11710, 105384)
	(train_half_length, test_half_length) = (int(val_length/2), int(test_length/2))
	
	#Generate artificial Z
	print('\n##############VALIDATION DATA#################')
	(TL_val, Z_val) = Z_new_generator(val_length, classifier_stats_val)
	print('\n##############TEST DATA#################')
	(TL_test, Z_test) = Z_new_generator(test_length, classifier_stats_test)
	
	#Calculate Ranking
	vr = classifier_rank_estimator(Z_val)

	#SML on validation data
	print(f'\n\n****VALIDATION SML RESULTS*****')
	y = SML(vr, Z_val)
	true_positives = sum(y[train_half_length:]==-1)
	false_negatives = sum(y[train_half_length:]==1)
	false_positives = sum(y[0:train_half_length]==-1)
	true_negatives = sum(y[0:train_half_length]==1)
	MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)

	#SML on test data
	print(f'\n\n****TEST SML RESULTS*****')
	y = SML(vr, Z_test)
	true_positives = sum(y[test_half_length:]==-1)
	false_negatives = sum(y[test_half_length:]==1)
	false_positives = sum(y[0:test_half_length]==-1)
	true_negatives = sum(y[0:test_half_length]==1)
	MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives):
	'''
	Print confusion matrix and other statistics
	'''
	total_samples = true_positives + false_positives + false_negatives + true_negatives
	test_accuracy = (true_positives+true_negatives)/total_samples
	test_precision = true_positives/(true_positives+false_positives)
	true_positive_rate = true_positives/(true_positives+false_negatives)
	false_positive_rate = false_positives/(false_positives+true_negatives)
	false_negative_rate = false_negatives/(true_positives+false_negatives)
	true_negative_rate = true_negatives/(true_negatives+false_positives)

	print(tabulate([['Predicted Anomalous', true_positives, false_positives], \
					['Predicted Normal', false_negatives, true_negatives]], \
			headers=[' ', 'Actual Anomalous', 'Actual Normal'], tablefmt='orgtbl'))
	print('Accuracy - ', test_accuracy)
	print('True Positive Rate - ', true_positive_rate)
	print('False Positive Rate - ', false_positive_rate)
	print('False Negative Rate - ', false_negative_rate)
	print('True Negative Rate - ', true_negative_rate)
	print('Balanced Accuracy - ', 0.5*(true_positive_rate+true_negative_rate))

#---------------------------------------------------------------------#
#######################################################################

#######################################################################
#---------------------------------------------------------------------#
def ensemble_stat_analysis():
	'''
	Statistical analysis of the ensemble data
	'''
	def stats_test(table):
		#Combinations on Validation Set
		combos = list(itertools.combinations(enumerate(table.T), 2))
		for combo in combos:
			(i, j) = (combo[0], combo[1])
			cl_tab = np.array(list([i[1], j[1]])).T
			cont_table = np.zeros((2,2))
			cont_table[0,0] = sum(cl_tab[:,0] & cl_tab[:,1])		#1 correct, 2 correct
			cont_table[0,1] = sum(cl_tab[:,0] & ~cl_tab[:,1])		#1 correct, 2 incorrect
			cont_table[1,0] = sum(~cl_tab[:,0] & cl_tab[:,1])		#1 incorrect, 2 correct
			cont_table[1,1] = sum(~cl_tab[:,0] & ~cl_tab[:,1])		#1 incorrect, 2 incorrect
			print('\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
			print(f'Classifiers being compared - {i[0]}, {j[0]}')
			print('\n', cont_table)
			
			#McNemar Test
			# calculate mcnemar test
			result = mcnemar(cont_table, exact=False, correction=True)
			# summarize the finding
			print('\n---------- McNemar Test -----------')
			print(f'Statistic = {result.statistic}, p-value = {result.pvalue}')
			# interpret the p-value
			alpha = 0.01
			if result.pvalue < alpha:
				print('Classifiers have different proportions of errors (reject H0)')
			else:
				print('Classifiers have same proportions of errors (fail to reject H0)')
		
			#Chi-square Test
			print('\n---------- Chi-Square Test ----------')
			(chi2, p, dof, ex) = chi2_contingency(cont_table, correction=False)
			print(f'Statistic = {chi2}, p-value = {p}')
			# interpret the p-value
			alpha = 0.01
			if p < alpha:
				print('Dependent(Reject H0)')
			else:
				print('Independent(fail to reject H0)')
			
	if REAL_DATA_ANALYSIS==True:
		print('\nEVALUATING ON REAL DATA--------------\n')
		validation_all = np.loadtxt(ENSEMBLE_TRAIN_DATA, delimiter=',').astype(int)
		test_all = np.loadtxt(ENSEMBLE_TEST_DATA, delimiter=',').astype(int)
	elif REAL_DATA_ANALYSIS==False:
		print('\nEVALUATING ON ARTIFICIAL DATA--------------\n')
		print('\n##############VALIDATION DATA#################')
		(tempA, validation_all) = Z_new_generator(11710, classifier_stats_val)
		print('\n##############TEST DATA#################')
		(tempB, test_all) = Z_new_generator(105384, classifier_stats_test)
	
	#Get metrics
	m_val = validation_all.shape[1]
	m_test = test_all.shape[1]
	val_half_len = int(validation_all.shape[0]/2)
	test_half_len = int(test_all.shape[0]/2)
	
	#Define labels
	val_TL = np.ones((validation_all.shape[0],m_val)).astype(int)
	val_TL[val_half_len:,:] = -1
	test_TL = np.ones((test_all.shape[0],m_test)).astype(int)
	test_TL[test_half_len:,:] = -1
	
	val_table = (validation_all==val_TL)
	test_table = (test_all==test_TL)
	
	#Statistical Tests
	print('\n\n################ Validation Data #################')
	print(f'Validation Covariance \n{np.cov(validation_all.T)}')
	stats_test(val_table)
	print('\n\n################ Test Data #################')
	print(f'Test Covariance \n{np.cov(test_all.T)}')
	stats_test(test_table)
	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#---------------------------------------------------------------------#
#MAIN
#---------------------------------------------------------------------#

'''
To evaluate performance of SML and RBM on artificial data
'''
#SML_new()
#RBM_new()				

'''
To evaluate ensemble performance from real predicted data
'''
#ml_analysis()

'''
To perform statistical analysis on ensemble data Real OR Artificial
'''
ensemble_stat_analysis()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
