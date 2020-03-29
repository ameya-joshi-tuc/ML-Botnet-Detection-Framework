#!/usr/bin/python

#---------------------------------------------------------------------#
#Ameya G. Joshi
#Professorship of Communication Networks
#TU Chemnitz
#Machine Learning based Botnet Detection Framework for SDN
#---------------------------------------------------------------------#

from __future__ import division
from random import *
from math import *

import threading
import queue as Queue
import requests
from requests.auth import HTTPBasicAuth
import time
import sys, os
import copy

import csv
from operator import itemgetter
from itertools import islice
import datetime

import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.neural_network import BernoulliRBM
from joblib import dump, load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses,regularizers
from tensorflow.keras.layers import Lambda, Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 17.4})

from tabulate import tabulate

#---------------------------------------------------------------------#
#Global Variables
#(We only need to use global keyword in a function if we want to do assignments/change them.
#Global is not needed for printing and accessing)
#---------------------------------------------------------------------#

#Save Models
SAVE_DATA = False

#FILE NAMES
NORMAL_TRAIN_DATA = 'normal_train.csv'			#Normal training data
ANOMALOUS_DATA = 'anomalous_test.csv'			#Anomalous validation/testing data
NORMAL_TEST_DATA = 'normal_test.csv'		#Normal validation/testing data

#Define all the required features here and in final_flow_data
raw_flow_data = [['Proto', 'Src_IP', 'Dst_IP', 'Src_Port', 'Dst_Port', 'Duration', 'N_pkts', 'Bytes', 'Last_seen', \
	'Pkts/sec', 'Bytes/sec', 'Flow IAT', 'Node IAT', 'Flow Direction', 'TW_FlowCount', 'TW_Node_SentReqCount', 'TW_Node_SentRespCount', \
	'Avg_pkts/sec', 'Avg_bytes/sec', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Node_IAT_Mean', 'Node_IAT_Std', 'Predicted Label', 'Actual Label']]

#Final list of lists to store generated flow data with additional required features to .csv file
final_flow_data = [['Proto', 'Src_IP', 'Dst_IP', 'Src_Port', 'Dst_Port', 'Duration', 'N_pkts', 'Bytes', 'Last_seen', \
	'Pkts/sec', 'Bytes/sec', 'Flow IAT', 'Node IAT', 'Flow Direction', 'TW_FlowCount', 'TW_Node_SentReqCount', 'TW_Node_SentRespCount', \
	'Avg_pkts/sec', 'Avg_bytes/sec', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Node_IAT_Mean', 'Node_IAT_Std', 'Predicted Label', 'Actual Label']]

#Suspicious list based on M out of N flows
#E.g. if Last 3 out of 5 flows between a source and destination are predicted anomalous, mark the IP addresses as suspicious
suspicious_list = []
M = 2		#Consecutive malicious flows to categorise a pair of IP Addresses as malicious
N = 3		#Window size of flows out of which to categorise

#Indices of features required by Machine Learning Algorithm
#ALL_REQUIRED_FEATURES = [5, 9, 10, 11, 13]						#Feature Set 1
#ALL_REQUIRED_FEATURES = [5, 9, 10, 11, 13, 17, 18, 19]			#Feature Set 2
#ALL_REQUIRED_FEATURES = [5, 9, 10, 11, 13, 17, 18, 19, 21, 12]	#Feature Set 3

ALL_REQUIRED_FEATURES = np.array(np.arange(21))
FEATURE_LENGTH = len(ALL_REQUIRED_FEATURES)

FLOW_TIMEOUT = 30			#Timeout in seconds; If flow exceeds this value, send to Machine Learning Thread
T_OBSERVATION = 3			#Time to collect statistics from switch in seconds
TIME_WINDOW = 120			#Time-window in seconds for calculating number of prior requests, responses etc
temp_flow_data = [[]]		#Temporary buffer to prevent duplicate entries on completed raw flows

#Scaling Type; 
#SCALE_TPE: 1-Normalisation, 2-Min-Max Scaling, 3-Yeo Johnson Transform
SCALE_TYPE = 3

#Machine Learning Model Choice;	
#MODEL_CHOICE: 1-OCSVM, 2-LOF, 3-Autoencoder/VAE, 4-Ensemble of OCSVM and LOF, 5-Ensemble of OCSVM,LOF and AE/VAE, 6 -KDE
MODEL_CHOICE = 6

#One Class SVM
(NU_VAL, GAMMA_VAL) = (0.1, 3.9)

#Local Outlier Factor
(CONTAMINATION_FACTOR, NUM_NEIGHBOURS) = (0.09, 20)			

#Kernel Density Estimation
(KDE_BANDWIDTH, KDE_THRESHOLD) = (0.21, 0.9)

#Autoencoder 
#AE_TYPE: 1- Vanilla Autoencoder, 2- VAE ; LAYER_CONF: 1- Deep AE , 2- Shallow AE
(AE_TYPE, LAYER_CONF) = (1, 1)
(INPUT_HIDDEN_ONE_SIZE, INPUT_HIDDEN_TWO_SIZE, LATENT_LAYER_SIZE, OUTPUT_HIDDEN_ONE_SIZE, OUTPUT_HIDDEN_TWO_SIZE) = (10,8,5,8,10)
(REG_PARAM, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT) = (0.0001, 128, 50, 0.2)

#Thresholds
(PERCENTILE_LOWER_BOUND, PERCENTILE_UPPER_BOUND) = (0, 90)

#RBM
SUSPICIOUS_MODE = False
(N_COMPONENTS, N_ITERATIONS, RBM_LEARNING_RATE, RBM_BATCH_SIZE) = (1, 100, 0.001, 10)

#Ensemble Threshold Percentages; Anomalous<=T1, T1<suspicious<=T2, T2<Normal (Give in decimal form, e.g. 0.7)
T1 = 0.5	#Global Threshold when SUSPICIOUS_MODE is False
T2 = 0.6

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#Global Functions
#---------------------------------------------------------------------#

#Start Statistic Collector thread
def StartStatisticCollector():
	#Start Statstic Collector thread for switch of:0000000000000001
	Stats = StatisticCollector(1,'of:0000000000000001')
	Stats.start()

#Start RawFlowExtractor
def StartRawFlowExtractor():
	#Start Raw Flow Extractor
	RawFlowExtr = RawFlowExtractor(2)
	RawFlowExtr.start()

#Start Raw_flow_data thread
def StartRawFlowProcessor():
	#Start Statstic Collector thread for switch of:0000000000000001
	RawFlowProc = RawFlowProcessor(3)
	RawFlowProc.start()

#Start Data Processor thread
def StartFinalDataProcessor():
	#Start Data Processing thread for calculating additional statistics for data extracted from Statistic Collector
	FinalDataProc = FinalDataProcessor(4)
	FinalDataProc.start()

#Start Machine Learning Algorithm thread
def StartMachineLearningAlgorithm(ml_choice):
	#Start Machine Learning Algorithm thread
	MLThread = MachineLearningAlgorithm(5, ml_choice)
	MLThread.start()

#---------------------------------------------------------------------#
#Threads
#---------------------------------------------------------------------#

#Statistic Collector Agent takes statistics of a switch from the SDN controller using REST APIs
class StatisticCollector(threading.Thread):
	def __init__(self, threadID, DeviceID):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.DeviceID = DeviceID  #of:000000000000000X - DeviceID
	
	def run(self):
		while True:
			try:
				url = 'http://134.109.4.70:8181/onos/v1/flows/' + self.DeviceID
				headers  = {"Accept": "application/json"}
				response = requests.get(url, headers = headers, auth=HTTPBasicAuth('karaf', 'karaf'))
				if response.json().has_key('flows'):
					for f in response.json()['flows']:
						if f['appId']=='org.onosproject.core':		
							pass
						#Flows from forwarding application
						if f['appId']!='org.onosproject.core':		
							q_statsCollector.put(f)					#Put to queue of next thread
				
			except Exception as ex:
				template = "An exception of type {0} occured. Arguments:\n{1!r}"
				message = template.format(type(ex).__name__, ex.args)
				print(message)
				print('\n StatisticCollector | There is an error... Exited on ',self.DeviceID)
				break
			time.sleep(T_OBSERVATION)
#---------------------------------------------------------------------#

#RawFlowExtractor extracts required attributes from each flow rule entry
#Flow Parser
class RawFlowExtractor(threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
	def run(self):
		global raw_flow_data
		while True:
			try:
				if not q_statsCollector.empty():
					f = q_statsCollector.get()
					criteria_list = f['selector']['criteria']
					current_flow_data = [0]*len(raw_flow_data[0])
					icmp_type = 0
					for g in criteria_list:
						if g['type']=='IP_PROTO':
							current_flow_data[0]=(g['protocol'])
						if g['type']=='IPV4_SRC':
							current_flow_data[1]=(g['ip'])
						if g['type']=='IPV4_DST':
							current_flow_data[2]=(g['ip'])
						if g['type']=='TCP_SRC':
							current_flow_data[3]=(g['tcpPort'])
						elif g['type']=='UDP_SRC':
							current_flow_data[3]=(g['udpPort'])
						elif g['type']=='ICMPV4_TYPE':
							current_flow_data[3]=0		#No source port
							current_flow_data[4]=0		#No destination port
							icmp_type = g['icmpType']	#If 8 it is a request, if 0 it is a response
							if icmp_type==0:
								current_flow_data[13] = 1
						if g['type']=='TCP_DST':
							current_flow_data[4]=(g['tcpPort'])
						elif g['type']=='UDP_DST':
							current_flow_data[4]=(g['udpPort'])
						
					current_flow_data[5:9] = [f['life'], f['packets'], f['bytes'], f['lastSeen']]
					q_rawData.put(current_flow_data)	#Update raw_flow_data entries based on changing flow data
					
			except Exception as ex:
				template = "An exception of type {0} occured. Arguments:\n{1!r}"
				message = template.format(type(ex).__name__, ex.args)
				print(message)
				print('\n RawFlowExtractor | There is an error... Exited on ',self.DeviceID)
				break
#---------------------------------------------------------------------#
		
#Raw Flow Processor processes incoming parsed flows from RawFlowExtractor and sends completed flows to Data Processor
#Individual Flow Aggregator
class RawFlowProcessor(threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
	
	def run(self):
		while True:
			global raw_flow_data
			try:
				if not q_rawData.empty():
					start_time = time.time()
					current_flow_data = q_rawData.get()
					count = 0
					for items in reversed(raw_flow_data):
						#If matching flow is found
						if (current_flow_data[0:5]==items[0:5]):
							count = 1
							#If Flow bytes and/or packets are being updated
							if (current_flow_data[6]!=items[6] or current_flow_data[7]!=items[7]
								):
								current_flow_data[8]=items[8]			#Retain timestamp for the first time the flow was seen
								if (current_flow_data[5]>FLOW_TIMEOUT):	#If a Flow remains in Flow Table for more than FLOW_TIMEOUT
									q_finalData.put(current_flow_data)	
								items[5:8] = current_flow_data[5:8]		#Get the new values
								break
							#If Flow bytes and/or packets are not being updated and Flow is idle in the Flow Table
							elif (current_flow_data[6]==items[6] or current_flow_data[7]==items[7]):
								if (current_flow_data[5]!=0 or current_flow_data[6]!=0):
									temp_value = items[5] + 3				#To account for system delays
									if (current_flow_data[5]>temp_value):	
										q_finalData.put(items)
										break
								elif (current_flow_data[5]==0 or current_flow_data[6]==0):	#Flow not yet updated
									break
					if count==0:	#New Flow Entry
						raw_flow_data.append(current_flow_data)
						raw_flow_data = sorted(raw_flow_data, key=itemgetter(8))	#Sort raw_flow_data according to last_seen
					
			except Exception as ex:
				template = "An exception of type {0} occured. Arguments:\n{1!r}"
				message = template.format(type(ex).__name__, ex.args)
				print(message)
				print('\nRaw Flow Processor shows an error!!')
				break		
#---------------------------------------------------------------------#

#Final Data Processor calculates additional features from Raw Flow Data which will be used by the ML Algorithm
#Additional Feature Calculator
class FinalDataProcessor(threading.Thread):
	def __init__(self, threadID):
		threading.Thread.__init__(self)
		self.threadID = threadID
	
	#To calculate additional features
	def additional_temporal_features(self, processed_flow_data, local_idx, original_flow_data, local_flow_data):
		flow_iat_count = 0
		node_iat_count = 0
		flow_direction_count = 0
		local_window = 0
		
		def mean_calculation(local_flow_data_val, running_sum, iat_sum):
			running_sum = running_sum + local_flow_data_val
			iat_sum.append(local_flow_data_val)
			iat_sum = np.array(iat_sum)
			iat_mean = np.mean(iat_sum)
			iat_std = np.std(iat_sum)
			
			return (iat_mean, iat_std)
		
		#IMPORTANT NOTE - Will not happen normally, but if the threads are processed out-of-order, the IAT
		#times may evaluate to negative. Hence a check is made and if negative, it finds the next flow instead
		
		#IMPORTANT NOTE - ONOS Reactive Forwarding App gives Timestamps in ms which can be seen in LastSeen
		#Sometimes both request and response flows have the same Timestamps and it is not possible to say which flow entry is the
		#request and which flow entry is the response which is crucial for deciding direction. A possible improvement could be to improve
		#resolution of ONOS app to us instead of ms, or use a different better Forwarding Application
		
		#To calculate IATs and Flow Direction
		for stuff in islice(reversed(processed_flow_data), local_idx, None):
			if original_flow_data!=stuff:
				#To calculate flow inter-arrival time if similar flow between same source and destination
				if (	(local_flow_data[0:3]==stuff[0:3]) and 
						(local_flow_data[3]!=stuff[3] or local_flow_data[4]!=stuff[4]) and
						(flow_iat_count==0)
					):
					flow_value = 0.001*(local_flow_data[8]-stuff[8])	#Last seen is in ms
					if flow_value>=0:
						flow_iat_count = 1
						local_flow_data[11] = flow_value
				
				#To calculate node inter-arrival time for flow from same source
				if (	(local_flow_data[0:2]==stuff[0:2]) and
						(node_iat_count==0)
					):
					node_value = 0.001*(local_flow_data[8]-stuff[8])	#Last seen is in ms
					if node_value>=0:
						node_iat_count = 1
						local_flow_data[12] = node_value
				
				#To find out the direction
				if (	(local_flow_data[0]==stuff[0]) and 
						(local_flow_data[1]==stuff[2]) and 
						(local_flow_data[2]==stuff[1]) and
						(local_flow_data[3]==stuff[4]) and 
						(local_flow_data[4]==stuff[3]) and
						(flow_direction_count==0)
					):
					if local_flow_data[8]>stuff[8]:
						local_flow_data[13] = 1
						flow_direction_count = 1
					elif local_flow_data[8]==stuff[8]:
						if (	((local_flow_data[3]==80) or (local_flow_data[3]==8080) or (local_flow_data[3]==8081) or (local_flow_data[3]==8082)) and 	#HTTP or Ares Server Port or BYOB Port
								(local_flow_data[0]==6)
							):	
							local_flow_data[13] = 1
							flow_direction_count = 1
				elif (	((local_flow_data[3]==80) or (local_flow_data[3]==8080) or (local_flow_data[3]==8081) or (local_flow_data[3]==8082)) and 			#HTTP or Ares Server Port
						(local_flow_data[0]==6)	
					):
					local_flow_data[13] = 1
					flow_direction_count = 1
				#If all 3 obtained
				if (	(flow_iat_count==1) and
						(node_iat_count==1) and
						(flow_direction_count==1)
					):
					break
		
		running_sum_flow = 0
		running_sum_node = 0
		running_sum_pkts = 0
		running_sum_bytes = 0
		flow_iat_sum = []
		node_iat_sum = []
		pkts_sum = []
		bytes_sum = []
		
		#To calculate Time-window statistics
		for stuff in islice(reversed(processed_flow_data), local_idx, None):
			if ((original_flow_data!=stuff) and (stuff[0]!='Proto')):
				local_window = int(stuff[8])+(1000*TIME_WINDOW)
				
				if (local_flow_data[8]<=local_window):	#Check for previous Time-Window
					
					#Features between same source and destination in defined Time-Window
					if (local_flow_data[0:3]==stuff[0:3]):
						local_flow_data[14]=int(local_flow_data[14])+1
						
						#Pkts sum
						running_sum_pkts = running_sum_pkts + stuff[9]
						pkts_sum.append(stuff[9])
						#Bytes sum
						running_sum_bytes = running_sum_bytes + stuff[10]
						bytes_sum.append(stuff[10])
						#Flow IAT sum
						running_sum_flow = running_sum_flow + stuff[11]
						flow_iat_sum.append(stuff[11])
						
					#Features considering same source
					if (local_flow_data[0:2]==stuff[0:2]):
						#Number of requests sent by same node in defined Time-Window
						if local_flow_data[13]==0:
							local_flow_data[15]=int(local_flow_data[15])+1
						#Number of responses sent by same node in defined Time-Window
						elif local_flow_data[13]==1:
							local_flow_data[16]=int(local_flow_data[16])+1
						#Node IAT sum
						running_sum_node = running_sum_node + stuff[12]
						node_iat_sum.append(stuff[12])
				else:
					break
				
		(pkts_mean, pkts_std) = mean_calculation(local_flow_data[9], running_sum_pkts, pkts_sum)
		(bytes_mean, bytes_std) = mean_calculation(local_flow_data[10], running_sum_bytes, bytes_sum)
		(flow_iat_mean, flow_iat_std) = mean_calculation(local_flow_data[11], running_sum_flow, flow_iat_sum)
		(node_iat_mean, node_iat_std) = mean_calculation(local_flow_data[12], running_sum_node, node_iat_sum)
		
		local_flow_data[17] = pkts_mean
		local_flow_data[18] = bytes_mean
		local_flow_data[19] = flow_iat_mean
		local_flow_data[20] = flow_iat_std
		local_flow_data[21] = node_iat_mean
		local_flow_data[22] = node_iat_std
						
	def run(self):
		global temp_flow_data
		global final_flow_data
		while True:
			try:
				if not q_finalData.empty():
					start_time = time.time()
					local_flow_data = q_finalData.get()
					if local_flow_data not in temp_flow_data:
						processed_flow_data = raw_flow_data[:]
						temp_flow_data.append(local_flow_data)
						original_flow_data = local_flow_data
						
						for idx, stuff in enumerate(reversed(processed_flow_data)):
							if original_flow_data==stuff:
								local_idx = idx
					
						if local_flow_data[5]!=0:
							local_flow_data[9] = float(local_flow_data[6])/float(local_flow_data[5])
							local_flow_data[10] = float(local_flow_data[7])/float(local_flow_data[5])
						else:
							local_flow_data[9] = 0
							local_flow_data[10] = 0
						
						#To calculate temporal features
						self.additional_temporal_features(processed_flow_data, local_idx, original_flow_data, local_flow_data)
						
						if local_flow_data not in final_flow_data:
							final_flow_data.append(local_flow_data)
							q_MLData.put(local_flow_data)
							final_flow_data = sorted(final_flow_data, key=itemgetter(8))
						
			except Exception as ex:
				template = "An exception of type {0} occured. Arguments:\n{1!r}"
				message = template.format(type(ex).__name__, ex.args)
				print(message)
				print('\nFinal Data Processor shows an error!!')
				break
#---------------------------------------------------------------------#

#MachineLearningAlgorithm trains the model using data from final_flow_data or is used for testing data offline or real-time
#Machine Learning based Predictor
class MachineLearningAlgorithm(threading.Thread):
	def __init__(self, threadID, MLChoice):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.MLChoice = MLChoice
	
	#Read contents of CSV file
	def read_csv_file(self, file_name):
		with open(file_name, 'r') as f:
			reader = csv.reader(f)
			data = list(reader)
		
		return data
	
	#https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
	def store_AE_threshold(self, thresholds, file_name):
		with open(file_name, "w") as f:
			for s in thresholds:
				f.write(str(s) +"\n")					# Store threshold values as text file
	
	#Load AE thresholds
	def load_AE_threshold(self, file_name):
		thresholds = []
		with open(file_name, "r") as f:
			for line in f:
				thresholds.append(float(line.strip()))	# Load stored threshold values
		return thresholds
	
	#Load stored ML Models
	def load_model_data(self):
		scaler = load('train_scaler.joblib')					#Load stored scaler
		if (MODEL_CHOICE==3 or MODEL_CHOICE==5):
			autoencoder = load_model('ae_model.h5') 			#Load the stored autoencoder Keras model
			latent_model = load_model('ae_latent_model.h5')	#Load the stored latent Keras model
		else:
			autoencoder=None
			latent_model=None
		clf = load('ocsvm_clf.joblib')							#Load stored OCSVM model
		lof = load('lof_clf.joblib')							#Load stored LOF model
		rbm = load('rbm_ensemble_all.joblib')						#Load stored RBM model
		
		return (scaler, clf, lof, autoencoder, latent_model, rbm)
	
	#Pre-processing to select required features, perform categorical encoding etc.
	def feature_extractor(self, data):
		global FEATURE_LENGTH
		temp_data =[[]]				
		for stuff in data:
			new_stuff = []
			for n in ALL_REQUIRED_FEATURES:
				if n==0:
					if int(stuff[n])==6:
						new_stuff.append(0)
					else:
						new_stuff.append(1)
				else:	
					new_stuff.append(float(stuff[n]))
			FEATURE_LENGTH = len(new_stuff)
			temp_data.append(new_stuff)
			del new_stuff
		del temp_data[0]
		
		return np.array(temp_data)
	
	def feature_extractor_upd(self, data):
		data = np.array(data).astype(float)
		data = np.delete(data, [8,11], axis=1)
		return data
	
	def scale_data(self, train_data, normal_test_data, anomalous_test_data):
		if SCALE_TYPE==1:
			scaler = preprocessing.StandardScaler().fit(train_data)						#Standardisation/Z-score Normalisation
		elif SCALE_TYPE==2:
			scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(train_data)	#Min-Max Scaling
		elif SCALE_TYPE==3:
			scaler = PowerTransformer().fit(train_data)									#Yeo-Johnson Transform
		
		#Scaled Data
		train_data = scaler.transform(train_data)
		normal_test_data = scaler.transform(normal_test_data)
		anomalous_test_data = scaler.transform(anomalous_test_data)
		
		return (scaler, train_data, normal_test_data, anomalous_test_data)

	#One-class Support Vector Machines
	#Reference - https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
	def ML_OCSVM(self, train_data, normal_test_data, anomalous_test_data):
		
		clf = svm.OneClassSVM(nu=NU_VAL, kernel="rbf", gamma=GAMMA_VAL)
		print(f'\nStarting OCSVM fit.... - {datetime.datetime.now().time()}')
		clf.fit(train_data)
		predict_normal_test = clf.predict(normal_test_data)
		predict_anomalous_test = clf.predict(anomalous_test_data)
		print(f'\nOCSVM prediction finished - {datetime.datetime.now().time()}')
		true_positives = predict_anomalous_test[predict_anomalous_test == -1].size
		false_positives = predict_normal_test[predict_normal_test == -1].size
		false_negatives = predict_anomalous_test[predict_anomalous_test == 1].size
		true_negatives = predict_normal_test[predict_normal_test == 1].size
		
		#Save Prediction Data
		if SAVE_DATA==True:
			total_predicted_data = np.concatenate((predict_normal_test, predict_anomalous_test), axis=0)
			total_predicted_data = np.append(total_predicted_data, (true_positives, false_negatives, false_positives, true_negatives), axis=0)
			temp_name = "ocsvm_test"+str(NU_VAL)+"_"+str(GAMMA_VAL)
			temp_name = temp_name.replace('.','-')
			np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
		
		return (true_positives, false_positives, false_negatives, true_negatives, clf)
	
	#Local Outlier Factor
	#Reference - https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html
	def ML_LOF(self, train_data, normal_test_data, anomalous_test_data):
		
		lof = LocalOutlierFactor(n_neighbors=NUM_NEIGHBOURS, contamination=CONTAMINATION_FACTOR, novelty=True)
		print(f'\nStarting LOF fit.... - {datetime.datetime.now().time()}')
		lof.fit(train_data)		
		predict_normal_test = lof.predict(normal_test_data)
		predict_anomalous_test = lof.predict(anomalous_test_data)
		print(f'\nLOF prediction finished.... - {datetime.datetime.now().time()}')
		true_positives = predict_anomalous_test[predict_anomalous_test == -1].size
		false_positives = predict_normal_test[predict_normal_test == -1].size
		false_negatives = predict_anomalous_test[predict_anomalous_test == 1].size
		true_negatives = predict_normal_test[predict_normal_test == 1].size
		
		#Save prediction Data
		if SAVE_DATA==True:
			total_predicted_data = np.concatenate((predict_normal_test, predict_anomalous_test), axis=0)
			total_predicted_data = np.append(total_predicted_data, (true_positives, false_negatives, false_positives, true_negatives), axis=0)
			temp_name = "lof_test"+str(CONTAMINATION_FACTOR)+"_"+str(NUM_NEIGHBOURS)
			temp_name = temp_name.replace('.','-')
			np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
		
		return (true_positives, false_positives, false_negatives, true_negatives, lof)
	
	#Kernel Density
	def ML_KDE(self, train_data, normal_test_data, anomalous_test_data):
		kde = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel='gaussian')
		print(f'\nStarting KDE fit.... - {datetime.datetime.now().time()}')
		kde.fit(train_data)
		print(f'\nStarting score_samples_train.... - {datetime.datetime.now().time()}')
		train_samples = kde.score_samples(train_data)
		threshold = np.percentile(train_samples, 100*(1-KDE_THRESHOLD))
		print(f'Threshold calculated - > {threshold}')
		
		#KDE Plot for Training Data
		# ~ plt.figure()
		# ~ sns.kdeplot(train_samples, label='Normal Train Sample Scores')
		# ~ plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
		# ~ plt.legend(loc='best')
		# ~ plt.title('Normal Train KDE Plot')
		# ~ plt.show(block=False)
		
		print(f'\nStarting normal_test_samples.... - {datetime.datetime.now().time()}')
		normal_test_samples = kde.score_samples(normal_test_data)
		print(f'\nStarting anomalous_test_samples.... - {datetime.datetime.now().time()}')
		anomalous_test_samples = kde.score_samples(anomalous_test_data)
		
		#KDE Plot for Test Data
		# ~ plt.figure()
		# ~ sns.kdeplot(normal_test_samples, label='Normal Test Sample Scores')
		# ~ sns.kdeplot(anomalous_test_samples, label='Anomalous Test Sample Scores')
		# ~ plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
		# ~ plt.legend(loc='best')
		# ~ plt.title('Test data KDE Plot')
		# ~ plt.show(block=False)
		
		#Convert to binary predictions based on calculated threshold
		predict_normal_test = np.where(normal_test_samples < threshold, -1, 1)
		predict_anomalous_test = np.where(anomalous_test_samples < threshold, -1, 1)
		
		#Display test results
		true_positives = predict_anomalous_test[predict_anomalous_test == -1].size
		false_positives = predict_normal_test[predict_normal_test == -1].size
		false_negatives = predict_anomalous_test[predict_anomalous_test == 1].size
		true_negatives = predict_normal_test[predict_normal_test == 1].size
		self.MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
		
		#Save prediction Data
		if SAVE_DATA==True:
			total_predicted_data = np.concatenate((predict_normal_test, predict_anomalous_test), axis=0)
			total_predicted_data = np.append(total_predicted_data, (true_positives, false_negatives, false_positives, true_negatives), axis=0)
			temp_name = "kde_test"+str(KDE_BANDWIDTH)+"_"+str(KDE_THRESHOLD)
			temp_name = temp_name.replace('.','-')
			np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
		
		return (true_positives, false_positives, false_negatives, true_negatives, kde)
		
	#Vanilla AutoEncoder
	#Reference - https://blog.keras.io/building-autoencoders-in-keras.html
	def ML_AE(self, train_data):
		# patient early stopping
		es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
		#Layer configuration, Shallow/Deep
		def activation_model():
			if LAYER_CONF==1:
				input_data = Input(shape=(train_data.shape[1],), name='input_data')
				inputHidden_1 = Dense(INPUT_HIDDEN_ONE_SIZE, activation='tanh', name='inputHidden_1', activity_regularizer=regularizers.l2(REG_PARAM))(input_data)
				inputHidden_2 = Dense(INPUT_HIDDEN_TWO_SIZE, activation='tanh', name='inputHidden_2', activity_regularizer=regularizers.l2(REG_PARAM))(inputHidden_1)
				latent_data = Dense(LATENT_LAYER_SIZE, activation='tanh', name='latent_data', activity_regularizer=regularizers.l2(REG_PARAM))(inputHidden_2)
				outputHidden_1 = Dense(OUTPUT_HIDDEN_ONE_SIZE, activation='tanh', name='outputHidden_1', activity_regularizer=regularizers.l2(REG_PARAM))(latent_data)
				outputHidden_2 = Dense(OUTPUT_HIDDEN_TWO_SIZE, activation='tanh', name='outputHidden_2', activity_regularizer=regularizers.l2(REG_PARAM))(outputHidden_1)
				output_data = Dense(train_data.shape[1], activation='linear', name='output_data', activity_regularizer=regularizers.l2(REG_PARAM))(outputHidden_2)
				return (input_data, latent_data, output_data)
			elif LAYER_CONF==2:
				input_data = Input(shape=(FEATURE_LENGTH,), name='input_data')
				latent_data = Dense(LATENT_LAYER_SIZE, activation='tanh', name='latent_data', kernel_regularizer=regularizers.l2(REG_PARAM))(input_data)
				output_data = Dense(FEATURE_LENGTH, activation='tanh', name='output_data', kernel_regularizer=regularizers.l2(REG_PARAM))(latent_data)
				return (input_data, latent_data, output_data)
		
		print(f'\nStarting AE operations.... - {datetime.datetime.now().time()}')
		(input_data, latent_data, output_data) = activation_model()
		autoencoder = Model(input_data, output_data)
		autoencoder.compile(optimizer='adam', loss='mean_squared_error')
		history = autoencoder.fit(train_data, train_data, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])
		latent_model = Model(autoencoder.input, autoencoder.get_layer('latent_data').output)
		
		return (history, autoencoder, latent_model)
	
	#Variational AutoEncoder
	#Reference - https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
	def ML_VAE(self, train_data):
		#VAE Sampling
		def VAEsampling(args):
			"""Reparameterization trick by sampling for an isotropic unit Gaussian.
			# Arguments
				args (tensor): mean and log of var of Q(z|X)
			# Returns
				z (tensor): sampled latent vector
			"""
			z_mean, z_log_var = args
			batch = K.shape(z_mean)[0]
			dim = K.int_shape(z_mean)[1]
			# by default, random_normal has mean=0 and std=1.0
			epsilon = K.random_normal(shape=(batch, dim))
			return z_mean + K.exp(0.5 * z_log_var) * epsilon
		
		#Use custom loss
		def VAELoss(z_mean, z_log_var):
			KL_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
			def lossFunction(input_data, output_data):
				mse_loss = K.mean(K.square(output_data - input_data), axis=-1)
				return mse_loss + KL_loss
			return lossFunction
		# patient early stopping
		es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
		
		#Layer configuration, Shallow/Deep
		if LAYER_CONF==1:
			input_data = Input(shape=(FEATURE_LENGTH,), name='input_data')
			inputHidden_1 = Dense(INPUT_HIDDEN_ONE_SIZE, activation='tanh', name='inputHidden_1', kernel_regularizer=regularizers.l2(REG_PARAM))(input_data)
			inputHidden_2 = Dense(INPUT_HIDDEN_TWO_SIZE, activation='tanh', name='inputHidden_2', kernel_regularizer=regularizers.l2(REG_PARAM))(inputHidden_1)
			latent_data = Dense(LATENT_LAYER_SIZE, activation='tanh', name='latent_data', kernel_regularizer=regularizers.l2(REG_PARAM))(inputHidden_2)		#latent_data is z_mean for VAE
			z_log_var = Dense(LATENT_LAYER_SIZE, activation='tanh', name='z_log_var', kernel_regularizer=regularizers.l2(REG_PARAM))(inputHidden_2)
			z = Lambda(VAEsampling, output_shape=(LATENT_LAYER_SIZE,), name='z')([latent_data, z_log_var])
			outputHidden_1 = Dense(OUTPUT_HIDDEN_ONE_SIZE, activation='tanh', name='outputHidden_1', kernel_regularizer=regularizers.l2(REG_PARAM))
			outputHidden_intermediate_1 = outputHidden_1(z)
			outputHidden_2 = Dense(OUTPUT_HIDDEN_TWO_SIZE, activation='tanh', name='outputHidden_2', kernel_regularizer=regularizers.l2(REG_PARAM))
			outputHidden_intermediate_2 = outputHidden_2(outputHidden_intermediate_1)
			output_data = Dense(FEATURE_LENGTH, activation='linear',name='vae_output', kernel_regularizer=regularizers.l2(REG_PARAM))
			output_vae = output_data(outputHidden_intermediate_2)
			vae = Model(input_data, output_vae, name='vae')
			vae.summary()
		elif LAYER_CONF==2:
			input_data = Input(shape=(FEATURE_LENGTH,), name='input_data')
			latent_data = Dense(LATENT_LAYER_SIZE, activation='tanh', name='latent_data', kernel_regularizer=regularizers.l2(REG_PARAM))(input_data)		#latent_data is z_mean for VAE
			z_log_var = Dense(LATENT_LAYER_SIZE, activation='tanh', name='z_log_var', kernel_regularizer=regularizers.l2(REG_PARAM))(input_data)
			z = Lambda(VAEsampling, output_shape=(LATENT_LAYER_SIZE,), name='z')([latent_data, z_log_var])
			output_data = Dense(FEATURE_LENGTH, activation='linear',name='vae_output', kernel_regularizer=regularizers.l2(REG_PARAM))
			output_vae = output_data(z)
			vae = Model(input_data, output_vae, name='vae')
			vae.summary()
		
		#Instantiate Encoder and Decoder separately
		#Encoder
		latent_model = Model(input_data, latent_data, name='encoder')
		#Decoder
		if LAYER_CONF==1:
			decoder_input = Input(shape=(LATENT_LAYER_SIZE,))
			decoderHidden_1 = outputHidden_1(decoder_input)
			decoderHidden_2 = outputHidden_2(decoderHidden_1)
			decoder_output = output_data(decoderHidden_2)
			decoder = Model(decoder_input, decoder_output, name='decoder')
			#Model
			test_output = decoder(latent_model(input_data))
			autoencoder = Model(input_data, test_output)
		elif LAYER_CONF==2:
			decoder_input = Input(shape=(LATENT_LAYER_SIZE,))
			decoder_output = output_data(decoder_input)
			decoder = Model(decoder_input, decoder_output, name='decoder')
			#Model
			test_output = decoder(latent_model(input_data))
			autoencoder = Model(input_data, test_output)
		#Train
		vae.compile(optimizer='adadelta', loss=VAELoss(latent_data, z_log_var))
		history = vae.fit(train_data, train_data, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])
		
		return (history, autoencoder, latent_model)
		
	def AE_threshold_calculation(self, autoencoder, train_data):
	
		train_predicted_data = autoencoder.predict(train_data)
		train_error = np.sum(np.square(train_predicted_data - train_data), axis=1)
		ae_threshold_upper = np.percentile(train_error[:], PERCENTILE_UPPER_BOUND)
		ae_threshold_lower = np.percentile(train_error[:], PERCENTILE_LOWER_BOUND)		
		
		return (ae_threshold_lower, ae_threshold_upper)
					
	def AE_stats_calculator(self, autoencoder, normal_test_data, anomalous_test_data, ae_threshold_lower, ae_threshold_upper):
		
		normal_test_reconstructed = autoencoder.predict(normal_test_data)
		normal_test_error = np.sum(np.square(normal_test_reconstructed - normal_test_data), axis=1)
		anomalous_test_reconstructed = autoencoder.predict(anomalous_test_data)
		anomalous_test_error = np.sum(np.square(anomalous_test_reconstructed - anomalous_test_data), axis=1)
		false_positives = normal_test_reconstructed[(normal_test_error>ae_threshold_upper)|(normal_test_error<ae_threshold_lower)]
		true_negatives = normal_test_reconstructed[(normal_test_error<ae_threshold_upper)&(normal_test_error>ae_threshold_lower)]
		true_positives = anomalous_test_reconstructed[(anomalous_test_error>ae_threshold_upper)|(anomalous_test_error<ae_threshold_lower)]
		false_negatives = anomalous_test_reconstructed[(anomalous_test_error<ae_threshold_upper)&(anomalous_test_error>ae_threshold_lower)]

		normal_test_error[(normal_test_error<ae_threshold_lower)|(normal_test_error>ae_threshold_upper)] = -1
		normal_test_error[(normal_test_error>=ae_threshold_lower)&(normal_test_error<=ae_threshold_upper)] = 1
		anomalous_test_error[(anomalous_test_error<ae_threshold_lower)|(anomalous_test_error>ae_threshold_upper)] = -1
		anomalous_test_error[(anomalous_test_error>=ae_threshold_lower)&(anomalous_test_error<=ae_threshold_upper)] = 1
		
		#Save prediction Data
		if SAVE_DATA==True:
			total_predicted_data = np.concatenate((normal_test_error, anomalous_test_error), axis=0)
			total_predicted_data = np.append(total_predicted_data, (true_positives.shape[0], false_negatives.shape[0], 
											false_positives.shape[0], true_negatives.shape[0]), axis=0)
			temp_name = 'ae_val'+str(INPUT_HIDDEN_ONE_SIZE)+str(INPUT_HIDDEN_TWO_SIZE)+str(LATENT_LAYER_SIZE) \
					+str(OUTPUT_HIDDEN_ONE_SIZE)+str(OUTPUT_HIDDEN_TWO_SIZE)
			np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
			print(f'\nAE evaluation finished.... - {datetime.datetime.now().time()}')
		
		return (true_positives, false_positives, false_negatives, true_negatives)
	
	def RBM_Ensemble(self, train_predict_all, anomalous_test_predict_all, normal_test_predict_all):
		
		print('Train Shape - ', train_predict_all.shape)
		print('Anomalous Shape -', anomalous_test_predict_all.shape)
		#Train RBM 
		rbm = BernoulliRBM(n_components = N_COMPONENTS, n_iter=N_ITERATIONS, learning_rate=RBM_LEARNING_RATE, batch_size=RBM_BATCH_SIZE)
		print(f'\nStarting RBM training.... {datetime.datetime.now().time()}')
		x = rbm.fit(train_predict_all)
		#Test RBM
		anomalous_test_probability = rbm.transform(anomalous_test_predict_all)
		normal_test_probability = rbm.transform(normal_test_predict_all)
		print(f'\nRBM complete.... - {datetime.datetime.now().time()}')
		if SAVE_DATA==True:
			dump(rbm, 'rbm_ensemble_newall.joblib', compress=True)	#Save RBM model
		
		if SUSPICIOUS_MODE==True:
			#Suspicious entries lie between the two thresholds
			print('Anomalous (TP) -', int((anomalous_test_probability[anomalous_test_probability<=T1].shape[0])/N_COMPONENTS))
			print('Possible Suspicious -', int((anomalous_test_probability[(T1<anomalous_test_probability)&(anomalous_test_probability<=T2)].shape[0])/N_COMPONENTS))
			print('Normal (FN)-', int((anomalous_test_probability[T2<anomalous_test_probability].shape[0])/N_COMPONENTS))
			print('Anomalous (FP) -', int((normal_test_probability[normal_test_probability<=T1].shape[0])/N_COMPONENTS))
			print('Possible Suspicious -', int((normal_test_probability[(T1<normal_test_probability)&(normal_test_probability<=T2)].shape[0])/N_COMPONENTS))
			print('Normal (TN)-', int((normal_test_probability[T2<normal_test_probability].shape[0])/N_COMPONENTS))
		else:
			true_positives = int((anomalous_test_probability[anomalous_test_probability<=T1].shape[0])/N_COMPONENTS)
			false_negatives = int((anomalous_test_probability[T1<anomalous_test_probability].shape[0])/N_COMPONENTS)
			false_positives = int((normal_test_probability[normal_test_probability<=T1].shape[0])/N_COMPONENTS)
			true_negatives = int((normal_test_probability[T1<normal_test_probability].shape[0])/N_COMPONENTS)
			self.MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
	
	def MLmodel_evaluation(self, true_positives, false_positives, false_negatives, true_negatives):
		total_samples = true_positives + false_positives + false_negatives + true_negatives
		test_accuracy = (true_positives+true_negatives)/total_samples
		test_precision = true_positives/(true_positives+false_positives)
		true_positive_rate = true_positives/(true_positives+false_negatives)
		false_positive_rate = false_positives/(false_positives+true_negatives)
		false_negative_rate = false_negatives/(true_positives+false_negatives)
		true_negative_rate = true_negatives/(true_negatives+false_positives)
		
		print('\n*** For normal test samples ***')
		print('Normal count - ', true_negatives, ' and Anomaly Count - ', false_positives)
		print('\n*** For anomalous test samples ***')
		print('Normal count - ', false_negatives, ' and Anomaly Count - ', true_positives)
		print(tabulate([['Predicted Anomalous', true_positives, false_positives], \
						['Predicted Normal', false_negatives, true_negatives]], \
				headers=[' ', 'Actual Anomalous', 'Actual Normal'], tablefmt='orgtbl'))
		print('True Positive Rate - ', true_positive_rate)
		print('False Positive Rate - ', false_positive_rate)
		print('False Negative Rate - ', false_negative_rate)
		print('True Negative Rate - ', true_negative_rate)
		print('Balanced Accuracy - ', 0.5*(true_positive_rate+true_negative_rate))
	
	def run(self):
		global final_flow_data
		try:
			if MLChoice==1:
				np.set_printoptions(suppress=True)
				
				print(f'Loading Data....- {datetime.datetime.now().time()}')
				
				#Normal Train Data
				data = self.read_csv_file(NORMAL_TRAIN_DATA)
				del data[0]
				train_data = self.feature_extractor_upd(data)
				
				#Anomalous Test Data
				data = self.read_csv_file(ANOMALOUS_DATA)
				#del data[0]
				anomalous_test_data = self.feature_extractor_upd(data)

				#Normal Test Data
				data = self.read_csv_file(NORMAL_TEST_DATA)
				#del data[0]
				normal_test_data = self.feature_extractor_upd(data)
				
				print(f'Data loaded!! - {datetime.datetime.now().time()}')
				
				#Scaling
				(scaler, train_data, normal_test_data, anomalous_test_data) = self.scale_data(train_data, normal_test_data, anomalous_test_data)
				if SAVE_DATA==True:
					dump(scaler, 'train_scaler.joblib', compress=True)	#Save Scaler
				
				#Only OCSVM
				if (MODEL_CHOICE==1 or MODEL_CHOICE==4 or MODEL_CHOICE==5):
					#OCSVM
					(true_positives, false_positives, false_negatives, true_negatives, clf) = self.ML_OCSVM(train_data, normal_test_data, anomalous_test_data)
					if SAVE_DATA==True:
						dump(clf, 'ocsvm_clf.joblib', compress=True)	#Save OCSVM model
				
				#Only LOF
				if (MODEL_CHOICE==2  or MODEL_CHOICE==4 or MODEL_CHOICE==5):
					#Local Outlier Factor
					(true_positives, false_positives, false_negatives, true_negatives, lof) = self.ML_LOF(train_data, normal_test_data, anomalous_test_data)
					if SAVE_DATA==True:
						dump(lof, 'lof_clf.joblib', compress=True)	#Save LOF model
				
				#Only KDE
				if (MODEL_CHOICE==6):
					(true_positives, false_positives, false_negatives, true_negatives, kde) = self.ML_KDE(train_data, normal_test_data, anomalous_test_data)
					if SAVE_DATA==True:
						dump(kde, 'kde_clf.joblib', compress=True)	#Save KDE model
				
				#To use AE Models
				if (MODEL_CHOICE==3 or MODEL_CHOICE==5):
					if AE_TYPE==1:
						#VanillaAutoencoder
						(history, autoencoder, latent_model) = self.ML_AE(train_data)
					elif AE_TYPE==2:
						#Variational AutoEncoder
						(history, autoencoder, latent_model) = self.ML_VAE(train_data)
					
					#Calculate Thresholds
					(ae_threshold_lower, ae_threshold_upper) = self.AE_threshold_calculation(autoencoder, train_data)
					thresholds = [ae_threshold_lower, ae_threshold_upper, train_data.shape[0]]
					(true_positives, false_positives, false_negatives, true_negatives) = \
						self.AE_stats_calculator(autoencoder, normal_test_data, anomalous_test_data, ae_threshold_lower, ae_threshold_upper)
					(false_positives, true_negatives, true_positives, false_negatives) = \
						(false_positives.shape[0], true_negatives.shape[0], true_positives.shape[0], false_negatives.shape[0])
					
					#Save Data to disk
					if SAVE_DATA==True:
						autoencoder.save('ae_model.h5')  			#Create an HDF5 file for the complete autoencoder Keras model
						latent_model.save('ae_latent_model.h5')		#Create an HDF5 file for the latent autoencoder Keras model
						self.store_AE_threshold(thresholds, file_name='ae_thresholds.txt')	#Store thresholds as text file
					
				if (MODEL_CHOICE==4 or MODEL_CHOICE==5):
					#Combine Training predictions of algorithms
					train_predict_ocsvm = clf.predict(train_data)
					train_predict_ocsvm[train_predict_ocsvm==-1]=0
					train_predict_ocsvm = np.expand_dims(train_predict_ocsvm, axis=1)
					train_predict_lof = np.array(lof.negative_outlier_factor_)
					train_predict_lof[train_predict_lof<np.percentile(train_predict_lof[:],100*CONTAMINATION_FACTOR)]=0
					train_predict_lof[train_predict_lof>np.percentile(train_predict_lof[:],100*CONTAMINATION_FACTOR)]=1
					train_predict_lof = np.expand_dims(train_predict_lof, axis=1)
					
					#Combine Anomalous Test predictions of alorithms
					anomalous_test_predict_ocsvm = clf.predict(anomalous_test_data)
					anomalous_test_predict_ocsvm[anomalous_test_predict_ocsvm==-1]=0					
					anomalous_test_predict_ocsvm = np.expand_dims(anomalous_test_predict_ocsvm, axis=1)					
					anomalous_test_predict_lof = lof.predict(anomalous_test_data)
					anomalous_test_predict_lof[anomalous_test_predict_lof==-1]=0
					anomalous_test_predict_lof = np.expand_dims(anomalous_test_predict_lof, axis=1)
					
					#Combine Normal Test predictions of algorithms
					normal_test_predict_ocsvm = clf.predict(normal_test_data)
					normal_test_predict_ocsvm = np.expand_dims(normal_test_predict_ocsvm, axis=1)
					normal_test_predict_ocsvm[normal_test_predict_ocsvm==-1]=0
					normal_test_predict_lof = lof.predict(normal_test_data)
					normal_test_predict_lof[normal_test_predict_lof==-1]=0
					normal_test_predict_lof = np.expand_dims(normal_test_predict_lof, axis=1)
					
					if MODEL_CHOICE==4:
						train_predict_all = np.concatenate((train_predict_ocsvm, train_predict_lof), axis=1)
						anomalous_test_predict_all = np.concatenate((anomalous_test_predict_ocsvm, anomalous_test_predict_lof), axis=1)
						normal_test_predict_all = np.concatenate((normal_test_predict_ocsvm, normal_test_predict_lof), axis=1)
					elif MODEL_CHOICE==5:
						train_predict_ae = (np.sum(np.square(autoencoder.predict(train_data) - train_data), axis=1) < ae_threshold_upper).astype(int)
						train_predict_ae = np.expand_dims(train_predict_ae, axis=1)
						train_predict_all = np.concatenate((train_predict_ocsvm, train_predict_lof, train_predict_ae), axis=1)
						anomalous_test_predict_ae = (np.sum(np.square(autoencoder.predict(anomalous_test_data) - anomalous_test_data), axis=1) < ae_threshold_upper).astype(int)
						anomalous_test_predict_ae = np.expand_dims(anomalous_test_predict_ae, axis=1)
						anomalous_test_predict_all = np.concatenate((anomalous_test_predict_ocsvm, anomalous_test_predict_lof, anomalous_test_predict_ae), axis=1)
						normal_test_predict_ae = (np.sum(np.square(autoencoder.predict(normal_test_data) - normal_test_data), axis=1) < ae_threshold_upper).astype(int)
						normal_test_predict_ae = np.expand_dims(normal_test_predict_ae, axis=1)
						normal_test_predict_all = np.concatenate((normal_test_predict_ocsvm, normal_test_predict_lof, normal_test_predict_ae), axis=1)
					
					self.RBM_Ensemble(train_predict_all, anomalous_test_predict_all, normal_test_predict_all)
				
				if ((MODEL_CHOICE==1) or (MODEL_CHOICE==2) or (MODEL_CHOICE==3) or (MODEL_CHOICE==6)):
					#Calculate Statistics
					self.MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
				
				q_endThread.put(1)
				
			if MLChoice==2:
				global suspicious_list
				
				def final_predictor(local_flow_data, local_prediction):
					flow_found_counter = 0
					local_flow_attribute = [local_flow_data[1], local_flow_data[2], local_prediction]
					print(local_flow_attribute)
					for data in suspicious_list:
						if local_flow_attribute[0:2]==data[0:2]:
							flow_found_counter = 1
							if len(data)>(N+2):
								del data[2]
								data.append(local_prediction)
							else:
								data.append(local_prediction)
							data_local = data[2:]
							if sum(data_local)>=M:
								print('\nAnomalous flow predicted between - {} ---> {}'.format(data[0],data[1]))
							break
						else:
							continue
					if flow_found_counter==0:
						suspicious_list.append(local_flow_attribute)
					print('Suspicious List - ', suspicious_list)
					
				#Load stored data
				(scaler, clf, lof, autoencoder, latent_model, rbm) = self.load_model_data()
				thresholds = self.load_AE_threshold(file_name='ae_thresholds.txt')
				ae_threshold_lower, ae_threshold_upper = thresholds[0], thresholds[1]
				train_data_size = thresholds[2]
				print('\nStored Model data successfully loaded')
				while True:
					try:
						if not q_MLData.empty():
							local_flow_data = q_MLData.get()
							feature_data = []
							local_prediction = 0
							for n in ALL_REQUIRED_FEATURES:
								feature_data.append(local_flow_data[n])
							if feature_data[0]==6:
								feature_data[0]=0
							else:
								feature_data[0]=1
							feature_data = np.array(feature_data)
							feature_data = np.reshape(feature_data,(1,-1))		#For one example
							feature_data = scaler.transform(feature_data)
							predict_ocsvm = clf.predict(feature_data)
							predict_lof = lof.predict(feature_data)
							if predict_ocsvm==-1:
								predict_ocsvm = 0
							if predict_lof==-1:
								predict_lof = 0
							predict_ocsvm = np.expand_dims(predict_ocsvm, axis=0)
							predict_lof = np.expand_dims(predict_lof, axis=0)
							predict_all = np.concatenate((predict_ocsvm, predict_lof), axis=0)
							predict_all = np.reshape(predict_all, (1,-1))
							predict_all = predict_all.transpose()
							predict_probability = rbm.transform(predict_all)
							
							if (predict_probability>T1):	#Normal Flow
								local_prediction = 0
								final_predictor(local_flow_data, local_prediction)
							elif (predict_probability<T1):	#Anomalous Flow
								local_prediction = 1
								final_predictor(local_flow_data, local_prediction)
							
							#Use for working Autoencoder architectures
							#~ predicted_local_flow_data = autoencoder.predict(feature_data)
							#~ predicted_error = np.sum(np.square(predicted_local_flow_data - feature_data), axis=1)
							#~ if (predicted_error<ae_threshold_upper and predicted_error>ae_threshold_lower):
								#~ local_prediction = 0
								#~ final_predictor(local_flow_data, local_prediction, predict_probability)
								#~ #Calculate Running Average Thresholds
								#~ if predicted_error>ae_threshold_mean:
									#~ ae_threshold_lower = (train_data_size*ae_threshold_lower + predicted_error)/(train_data_size+1)
									#~ ae_threshold_upper = (train_data_size*ae_threshold_upper + predicted_error)/(train_data_size+1)
								#~ elif predicted_error<ae_threshold_mean:
									#~ ae_threshold_lower = (train_data_size*ae_threshold_lower - predicted_error)/(train_data_size+1)
									#~ ae_threshold_upper = (train_data_size*ae_threshold_upper - predicted_error)/(train_data_size+1)
								#~ train_data_size = train_data_size + 1	
							#~ else:
								#~ local_prediction = 1
								#~ final_predictor(local_flow_data, local_prediction, predict_probability)
								
					except Exception as ex:
						template = "An exception of type {0} occured. Arguments:\n{1!r}"
						message = template.format(type(ex).__name__, ex.args)
						print(message)
						print('\nML Real-time Thread shows an error!!')
			
			if MLChoice==3:
				print(f'\n{datetime.datetime.now().time()} - Loading Machine Learning Models....\n\n')
				(scaler, clf, lof, autoencoder, latent_model, rbm) = self.load_model_data()
				if (MODEL_CHOICE==3 or MODEL_CHOICE==5):
					thresholds = self.load_AE_threshold(file_name='ae_thresholds.txt')
					ae_threshold_lower, ae_threshold_upper = thresholds[0], thresholds[1]
				print(f'\n{datetime.datetime.now().time()} - Stored Models successfully loaded')
				print('\n****************************************')
				print(f'\n{datetime.datetime.now().time()} - Loading data....')
				#Anomalous Test Data
				data = self.read_csv_file(ANOMALOUS_DATA)
				#del data[0]
				anomalous_test_data = self.feature_extractor_upd(data)
				
				#Normal Test Data
				data = self.read_csv_file(NORMAL_TEST_DATA)
				#del data[0]
				normal_test_data = self.feature_extractor_upd(data)
				print(f'\n{datetime.datetime.now().time()} - Data successfully loaded')
				print('\n****************************************')
				print(f'\n{datetime.datetime.now().time()} - Scaling Data......')
				#Scale Data
				normal_test_data = scaler.transform(normal_test_data)
				anomalous_test_data = scaler.transform(anomalous_test_data)
				print(f'\n{datetime.datetime.now().time()} - Scaler successfully applied')
				print('\n****************************************')
				print(f'\n{datetime.datetime.now().time()} - Beginning Machine Learning Predictions......')
				
				if MODEL_CHOICE==1:
					predict_normal_test = clf.predict(normal_test_data)
					predict_anomalous_test = clf.predict(anomalous_test_data)
					true_positives = predict_anomalous_test[predict_anomalous_test == -1].size
					false_positives = predict_normal_test[predict_normal_test == -1].size
					false_negatives = predict_anomalous_test[predict_anomalous_test == 1].size
					true_negatives = predict_normal_test[predict_normal_test == 1].size
					
					#Save Prediction Data
					if SAVE_DATA==True:
						total_predicted_data = np.concatenate((predict_normal_test, predict_anomalous_test), axis=0)
						total_predicted_data = np.append(total_predicted_data, (true_positives, false_negatives, false_positives, true_negatives), axis=0)
						temp_name = "ocsvm_val"+str(NU_VAL)+"_"+str(GAMMA_VAL)
						temp_name = temp_name.replace('.','-')
						np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
				
				elif MODEL_CHOICE==2: 
					predict_normal_test = lof.predict(normal_test_data)
					predict_anomalous_test = lof.predict(anomalous_test_data)
					true_positives = predict_anomalous_test[predict_anomalous_test == -1].size
					false_positives = predict_normal_test[predict_normal_test == -1].size
					false_negatives = predict_anomalous_test[predict_anomalous_test == 1].size
					true_negatives = predict_normal_test[predict_normal_test == 1].size
					
					#Save prediction Data
					if SAVE_DATA==True:
						total_predicted_data = np.concatenate((predict_normal_test, predict_anomalous_test), axis=0)
						total_predicted_data = np.append(total_predicted_data, (true_positives, false_negatives, false_positives, true_negatives), axis=0)
						temp_name = "lof_val"+str(CONTAMINATION_FACTOR)+"_"+str(NUM_NEIGHBOURS)
						temp_name = temp_name.replace('.','-')
						np.savetxt(temp_name+".csv", total_predicted_data, delimiter=",", fmt="%d")
				
				elif MODEL_CHOICE==3:
					(true_positives, false_positives, false_negatives, true_negatives) = \
							self.AE_stats_calculator(autoencoder, normal_test_data, anomalous_test_data, ae_threshold_lower, ae_threshold_upper)
					(false_positives, true_negatives, true_positives, false_negatives) = \
						(false_positives.shape[0], true_negatives.shape[0], true_positives.shape[0], false_negatives.shape[0])
				
				elif (MODEL_CHOICE==4 or MODEL_CHOICE==5):
					#Combine Anomalous Test predictions of alorithms
					anomalous_test_predict_ocsvm = clf.predict(anomalous_test_data)
					anomalous_test_predict_ocsvm[anomalous_test_predict_ocsvm==-1]=0					
					anomalous_test_predict_ocsvm = np.expand_dims(anomalous_test_predict_ocsvm, axis=1)					
					anomalous_test_predict_lof = lof.predict(anomalous_test_data)
					anomalous_test_predict_lof[anomalous_test_predict_lof==-1]=0
					anomalous_test_predict_lof = np.expand_dims(anomalous_test_predict_lof, axis=1)
					
					#Combine Normal Test predictions of algorithms
					normal_test_predict_ocsvm = clf.predict(normal_test_data)
					normal_test_predict_ocsvm = np.expand_dims(normal_test_predict_ocsvm, axis=1)
					normal_test_predict_ocsvm[normal_test_predict_ocsvm==-1]=0
					normal_test_predict_lof = lof.predict(normal_test_data)
					normal_test_predict_lof[normal_test_predict_lof==-1]=0
					normal_test_predict_lof = np.expand_dims(normal_test_predict_lof, axis=1)
					
					if MODEL_CHOICE==4:
						anomalous_test_predict_all = np.concatenate((anomalous_test_predict_ocsvm, anomalous_test_predict_lof), axis=1)
						normal_test_predict_all = np.concatenate((normal_test_predict_ocsvm, normal_test_predict_lof), axis=1)
					elif MODEL_CHOICE==5:
						anomalous_test_predict_ae = (np.sum(np.square(autoencoder.predict(anomalous_test_data) - anomalous_test_data), axis=1) < ae_threshold_upper).astype(int)
						anomalous_test_predict_ae = np.expand_dims(anomalous_test_predict_ae, axis=1)
						anomalous_test_predict_all = np.concatenate((anomalous_test_predict_ocsvm, anomalous_test_predict_lof, anomalous_test_predict_ae), axis=1)
						normal_test_predict_ae = (np.sum(np.square(autoencoder.predict(normal_test_data) - normal_test_data), axis=1) < ae_threshold_upper).astype(int)
						normal_test_predict_ae = np.expand_dims(normal_test_predict_ae, axis=1)
						normal_test_predict_all = np.concatenate((normal_test_predict_ocsvm, normal_test_predict_lof, normal_test_predict_ae), axis=1)
					
					#Test RBM
					anomalous_test_probability = rbm.transform(anomalous_test_predict_all)
					normal_test_probability = rbm.transform(normal_test_predict_all)
					
					if SUSPICIOUS_MODE==True:
						#Suspicious entries lie between the two thresholds
						print('Anomalous (TP) -', int((anomalous_test_probability[anomalous_test_probability<=T1].shape[0])/N_COMPONENTS))
						print('Possible Suspicious -', int((anomalous_test_probability[(T1<anomalous_test_probability)&(anomalous_test_probability<=T2)].shape[0])/N_COMPONENTS))
						print('Normal (FN)-', int((anomalous_test_probability[T2<anomalous_test_probability].shape[0])/N_COMPONENTS))
						print('Anomalous (FP) -', int((normal_test_probability[normal_test_probability<=T1].shape[0])/N_COMPONENTS))
						print('Possible Suspicious -', int((normal_test_probability[(T1<normal_test_probability)&(normal_test_probability<=T2)].shape[0])/N_COMPONENTS))
						print('Normal (TN)-', int((normal_test_probability[T2<normal_test_probability].shape[0])/N_COMPONENTS))
					else:
						true_positives = int((anomalous_test_probability[anomalous_test_probability<=T1].shape[0])/N_COMPONENTS)
						false_negatives = int((anomalous_test_probability[T1<anomalous_test_probability].shape[0])/N_COMPONENTS)
						false_positives = int((normal_test_probability[normal_test_probability<=T1].shape[0])/N_COMPONENTS)
						true_negatives = int((normal_test_probability[T1<normal_test_probability].shape[0])/N_COMPONENTS)
						
						print(f'\n{datetime.datetime.now().time()} - Predictions successful. Calculating statistics....')
						self.MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
								
				if ((MODEL_CHOICE==1) or (MODEL_CHOICE==2) or (MODEL_CHOICE==3)):
					#Calculate Statistics
					self.MLmodel_evaluation(true_positives, false_positives, false_negatives, true_negatives)
				
				print(f'\n{datetime.datetime.now().time()} - Terminating....')
				print('\n****************************************\n')
				
				q_endThread.put(1)
				
		except Exception as ex:
			template = "An exception of type {0} occured. Arguments:\n{1!r}"
			message = template.format(type(ex).__name__, ex.args)
			print(message)
			print('\nML Thread shows an error!!')

#---------------------------------------------------------------------#
#Main Thread
#---------------------------------------------------------------------#

#Define Queue for sharing data between threads
q_statsCollector = Queue.Queue()
q_rawData = Queue.Queue()
q_finalData = Queue.Queue()
q_MLData = Queue.Queue()
q_endThread = Queue.Queue()
collector_end = 0

print('.......Starting Program.......')
print('\nThe following Operation Modes are available -')
print('1) Data Collection - To collect data in a CSV file to use later for training Machine Learning(ML) model')
print('2) ML Training Mode - To train the ML model from an available CSV File')
print('3) Real-time Testing Mode - To test data in real-time on an already trained ML model')
print('4) Offline Testing Mode - To test data stored in a CSV File on an already trained ML model')

print('\nEnter the desired number to choose the mode of operation -')
while True:
	operation_mode = int(input('\n1 for Data Collection mode \n2 for ML Training Mode \n3 for Real-time Testing Mode \n4 for Offline Testing Mode \n-- '))
	if operation_mode not in (1, 2, 3, 4):
		print('\nEntry invalid. Please enter again!')
	else:
		print('\nOperation mode selected. Starting appropriate threads....')
		break

if operation_mode == 1:
	print('\n***Entered Data Collection Mode***')
	StartStatisticCollector()
	StartRawFlowExtractor()
	StartRawFlowProcessor()
	StartFinalDataProcessor()
	print('\nNow StatisticCollector and DataProcessor are ready!')
	print('\nAfter required data has been generated, press Ctrl+C to save output to CSV file and terminate the program')
	try:
		while True:
			pass
	except KeyboardInterrupt:
		
		print('\nUser terminated program. Writing data to csv file.......')
		#Move last header element to first position
		final_flow_data.insert(0,final_flow_data[-1])
		del final_flow_data[-1]
		
		#Save file as CSV
		file_name = input('\nEnter name of csv file without any spaces or extension type(e.g. Enter abc, NOT abc.csv) - ')
		with open(file_name + ".csv", "w", newline='') as f:
			writer = csv.writer(f)
			writer.writerows(final_flow_data)
		print('\nWrite operation completed. File saved as -', file_name+'.csv', 'in same folder')
		print('\nPress Ctrl+Z to terminate Program!!')
		collector_end = 1
		sys.exit(1)

elif operation_mode == 2:
	print('\n***Entered Machine Learning Training and Testing Mode***')
	MLChoice = 1	#For training Data
	StartMachineLearningAlgorithm(MLChoice)
	
elif operation_mode == 3:
	print('\n***Entered Real-time Testing Mode***')
	MLChoice = 2	#For testing Data in real-time
	StartStatisticCollector()
	StartRawFlowExtractor()
	StartRawFlowProcessor()
	StartFinalDataProcessor()
	StartMachineLearningAlgorithm(MLChoice)
	try:
		while True:
			pass
	except KeyboardInterrupt:
		print('\n******Code Exited******')
		sys.exit(1)
	
elif operation_mode == 4:
	print('\n***Entered Offline Testing Mode***')
	MLChoice = 3	#For testing Data offline
	StartMachineLearningAlgorithm(MLChoice)
	
else:
	print('\nAn Error occured. Program Execution Failed!!')

while True:
	if (collector_end==1):
		break
	if not q_endThread.empty():
		print("Ending Operation!!!!!!")
		break
