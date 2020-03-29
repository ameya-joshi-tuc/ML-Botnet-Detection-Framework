# ML-Botnet-Detection-Framework

Repository for a machine learning based botnet detection framework based on SDN written in Python3

# Files:

botnet_detection_framework.py -
Contains the main framework which can operate in offline and online(real-time) mode. The machine learning algorithms used are One Class Support Vector Machines, Local Outlier Factor, Autoencoders, Kernel Density Estimation and Restricted Boltzmann Machines

botnet_additional_ensemble.py -
Contains the ensemble approaches (namely the Restricted Boltzmann Machine and the Spectral Meta Learner) which combine individual binary predictions to provide a consensus. Also provides some techniques to perform a statistical data analysis for evaluating the classifier dependence

# References:

[1] Nicolau, Miguel, and James McDermott. "Learning neural representations for network anomaly detection." IEEE transactions on cybernetics 49.8 (2018): 3074-3087.

[2] Parisi, Fabio, et al. "Ranking and combining multiple predictors without labeled data." Proceedings of the National Academy of Sciences 111.4 (2014): 1253-1258.

[3] Shaham, Uri, et al. "A deep learning approach to unsupervised ensemble learning." International conference on machine learning. 2016.
