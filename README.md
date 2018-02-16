# Credit-Card-Fraud
Files to explore prediction of fraud with the following dataset: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

The dataset can be downloaded from: https://www.kaggle.com/dalpozz/creditcardfraud

The approach will be to use an autoencoder to understand learn about the dataset and to identify the transactions with the biggest error from the autoencoder reconstruction.

The autoencoder will then be used as pre-training for a neural network classifier

Further explanation of the file and approach will be added shortly

PLease note that for space reasons I have not included interim files that are needed between the train and classify activities, hence it will be necessary to run the fraud_demo_train file prior to the fraud_demo_classify one.

Useful information that helped provide guidance for the work was obtained from:
Venelin Valkov: https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd

Also Shirin Glander: https://shiring.github.io/machine_learning/2017/05/01/fraud

