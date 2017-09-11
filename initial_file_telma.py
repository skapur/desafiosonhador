
#===================
#     Packages
#===================

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


#===================
#     Options
#===================

# Set ipython's max row display
# pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 50
# pd.set_option('display.max_columns', 50)


#======================
#     Data Reading
#======================

def readDatasets (filenames, transpose = True):
    datasets = {}
    for filename in filenames:
        datasets[filename] = pd.read_csv(filename)

    # Join the datasets matching IDs
    keys = list(datasets.keys())
    result = pd.concat([datasets[keys[i]] for i in range(0, len(keys))],
                        axis = 1, join = 'inner').drop(['Unnamed: 0'], axis = 1)
    if transpose:
        return result.transpose()
    else:
        return result


def readClinicalData (file):
    file = pd.read_csv(file)

    return file


def getClinicalVector (patientList, clinDataset, clinField):
    # Get a vector from clinical data (to use as output variable)
    res = []
    for patient in patientList:
        res.append(clinDataset[clinDataset.Patient == patient][clinField].item())

    return res


#===========================
#        Dataset Info
#===========================

def datasetInfo (dataset, clinDataset = None):
    print('\n', '='*40, '\n', ' '*10, 'DATASET INFO \n', '='*40)

    print('Number of dataset samples: ', dataset.shape[0])
    print('Number of dataset features: ', dataset.shape[1])
    print('\n Sample names: \n', list(dataset.index))
    print('\n First 10 samples of the dataset: \n', dataset.iloc[0:9])

    if clinDataset is not None:
        print('\n', '='*40, '\n', ' '*10, 'METADATA INFO \n', '='*40)
        print('\n Metadata fields: \n', clinDataset.columns.values)
        print('\n First 20 samples of the dataset: \n', clinDataset.iloc[0:20, 0:9])


#===========================
#       Preprocessing
#===========================
def preprocess (dataset, method = 'scaler'):
    if method == 'scaler':
        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler().fit(dataset.values)
        newVals = scaler.transform(dataset.values)
    elif method == 'log':
        # Base 10 logarithm
        newVals = np.log10(dataset.values)
    else:
        print('Invalid method! Choose either "scaler" or "log".')

    return newVals


#===========================
#     Feature Selection
#===========================
def featureSelection(trainData, outputVector, method = "filter", percentile = 80, n = 154, model = "decisionTree"):
    if method == "filter":
        newDataset = SelectPercentile(f_classif, percentile).fit_transform(trainData, outputVector)
    elif method == "rfe":
        if model == "decisionTree": estimator = DecisionTreeClassifier()
        elif model == "logisticRegression": estimator = LogisticRegression()
        elif model == "svm": estimator = SVC()
        elif model == "rand_forest": estimator = RandomForestClassifier()
        newDataset = RFE(estimator, n).fit_transform(trainData, outputVector)

    return newDataset

#===============================================
#               Model Training
#===============================================

def modelTrain(values, outputVector, method = "knn", cv = 10, testVals = None):
    if method == "knn":
        estimator = KNeighborsClassifier(7)
    elif method == "nbayes":
        estimator = GaussianNB()
    elif method == "decisionTree":
        estimator = DecisionTreeClassifier(max_depth = 4, criterion = "gini", splitter = "random")
    elif method == "logisticRegression":
        estimator = LogisticRegression(solver = 'newton-cg', C = 1, penalty = "l2", tol = 0.001, multi_class = 'multinomial')
    elif method == "svm":
        estimator = SVC(kernel = "linear", C = 1, probability = True, gamma = 0.0001)
    elif method == 'nnet':
        estimator = MLPClassifier(solver = 'lbfgs', activation = "logistic", hidden_layer_sizes = (250,), alpha = 0.001)
    elif method == 'rand_forest':
        estimator = RandomForestClassifier(max_depth = 5, criterion = "entropy", n_estimators = 100)
    elif method == 'bagging':
        estimator = BaggingClassifier(max_samples = 1, bootstrap = True)

    if testVals is not None:
        model = estimator.fit(values, outputVector)
        return model.predict_proba(testVals)
    else:
        name = estimator.__class__.__name__
        mscore = ['accuracy', 'f1', 'neg_log_loss', 'recall']
        scores = cross_validate(estimator, values, outputVector, cv = cv, scoring = mscore)
        print("="*30)
        print(name)
        print('****Results****')
        print('Fit time: %0.4f (+/- %0.4f)' % (np.mean(scores['fit_time']), np.std(scores['fit_time'])))
        print('Score time: %0.4f (+/- %0.4f)' % (np.mean(scores['score_time']), np.std(scores['score_time'])))
        print('Test accuracy: %0.4f (+/- %0.4f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
        print('Train accuracy: %0.4f (+/- %0.4f)' % (np.mean(scores['train_accuracy']), np.std(scores['train_accuracy'])))
        print('Test F-score: %0.4f (+/- %0.4f)' % (np.mean(scores['test_f1']), np.std(scores['test_f1'])))
        print('Train F-score: %0.4f (+/- %0.4f)' % (np.mean(scores['train_f1']), np.std(scores['train_f1'])))
        print('Test log-loss: %0.4f (+/- %0.4f)' % (-np.mean(scores['test_neg_log_loss']), np.std(scores['test_neg_log_loss'])))
        print('Train log-loss: %0.4f (+/- %0.4f)' % (-np.mean(scores['train_neg_log_loss']), np.std(scores['train_neg_log_loss'])))
        print('Test recall: %0.4f (+/- %0.4f)' % (np.mean(scores['test_recall']), np.std(scores['test_recall'])))
        print('Train recall: %0.4f (+/- %0.4f)' % (np.mean(scores['train_recall']), np.std(scores['train_recall'])))
        print("="*30)


#===============================================
#   Model Training W/ Parameter Optimization
#===============================================

def modelTrainOptimization(values, outputVector, testVals = None, method = "knn", cross_val = 5, print_res = False):
    if method == "knn":
        estimator = KNeighborsClassifier(3)
        param = [{'n_neighbors': [1,2,3,4,5,6,7]}]
    elif method == "decisionTree":
        estimator = DecisionTreeClassifier()
        param = {'criterion': ['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150], 'splitter': ['random', 'best']}
    elif method == "logisticRegression":
        estimator = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')
        param = [{'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}]
    elif method == "svm":
        estimator = SVC(kernel = "linear", C = 0.025, probability = True)
        param = [{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'class_weight': [{1:3}, {1:4},{1:5}]}]
    elif method == 'nnet':
        estimator = MLPClassifier(solver = 'lbfgs')
        param = [{'hidden_layer_sizes': [(90,), (100,), (110,)],
                  'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [1e-5, 0.9]}]
    elif method == 'rand_forest':
        estimator = RandomForestClassifier()
        param = [{'n_estimators': [10, 20, 40, 60, 100], 'criterion': ['gini','entropy'],
                  'max_depth':[5,10,20,45,75,100,150]}]
    elif method == 'bagging':
        estimator = BaggingClassifier()
        param = [{'max_samples': [0.2, 0.5, 0.8, 1], 'bootstrap': [True, False]}]

    model = GridSearchCV(estimator, param_grid = param, scoring = 'neg_log_loss', refit = 'True', n_jobs = 1, cv = cross_val)
    model.fit(values, outputVector)

    if testVals is not None:
        return model.predict_proba(testVals)

    elif print_res:
        name = estimator.__class__.__name__
        print("="*30)
        print(name)
        print('****Results****')
        print("Best parameters set found on development set: \n", model.best_params_)
        print("Best score (log-loss): \n", abs(model.best_score_))
        print("="*30)





def testingMicroarrays (featSel = True, perc = 40, preprocessing = True, method = 'log'):
    # print(os.listdir('Microarray Data'))
    dataset = readDatasets(['D:\Dropbox\workspace_intelij\DREAM_Challenge\Microarray Data\EMTAB4032entrezIDlevel.csv',
                            'D:\Dropbox\workspace_intelij\DREAM_Challenge\Microarray Data\GSE19784HOVON65entrezIDlevel.csv',
                            'D:\Dropbox\workspace_intelij\DREAM_Challenge\Microarray Data\GSE24080UAMSentrezIDlevel.csv'],
                            transpose = True)

    # Metadata
    clin = readClinicalData('globalClinTraining.csv')
    clin = clin[clin.HR_FLAG != 'CENSORED'] # Remove patients with no progression

    nopatientdata = [patient for patient in list(dataset.index) if patient not in list(clin['Patient'])]
    dataset = dataset.drop(nopatientdata) # Remove patients with no metadata
    outputVector = getClinicalVector(list(dataset.index), clin, 'HR_FLAG')
    outputVector = np.array(outputVector) == 'TRUE'

    if preprocessing:
        trainData = preprocess(dataset, method = method) # with preprocessing
    else:
        trainData = dataset.values

    print('Original dataset dimensions: ', dataset.shape)

    if featSel:
        # Feature selection
        newTrainData = featureSelection(trainData, outputVector, percentile = perc)
        print('Filtered dataset dimensions: ', newTrainData.shape)
    else:
        newTrainData = trainData

    models = ['knn', 'nbayes', 'decisionTree', 'logisticRegression', 'svm',
              'nnet', 'rand_forest', 'bagging']

    # Test models
    for model in models:
        modelTrain(newTrainData, outputVector, method = model)

    # # Test models w/ parameter optimization
    # for model in models:
    #     if model is not 'nbayes':
    #         modelTrainOptimization(newTrainData, outputVector, method = model, print_res = True)

    # datasetInfo(dataset, clinDataset = clin)


def testingRNASeq (featSel = True, perc = 40, preprocessing = True, method = 'log'):
    genes = pd.read_table('D:\Dropbox\workspace_intelij\DREAM_Challenge\RNA Seq Data\MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_TPM.txt').drop(['GENE_ID'], axis = 1)
    transcripts = pd.read_table('D:\Dropbox\workspace_intelij\DREAM_Challenge\RNA Seq Data\MMRF_CoMMpass_IA9_E74GTF_Salmon_Transcript_TPM.txt').drop(['TRANSCRIPT_ID'], axis = 1)

    # Join datasets
    dataset = pd.concat([genes, transcripts], ignore_index = True).transpose()

    # Join duplicates by mean
    patients = ['_'.join(patient.split(sep = '_')[0:2]) for patient in list(dataset.index)]
    dataset.index = patients
    dataset.index.name = 'Patient'
    dataset = dataset.groupby(dataset.index.name)[dataset.columns.values].mean()

    # Metadata
    clin = readClinicalData('D:\Dropbox\workspace_intelij\DREAM_Challenge\Clinical Data\globalClinTraining.csv')
    clin = clin[clin.HR_FLAG != 'CENSORED'] # Remove patients with no progression

    #print(clin['HR_FLAG'].value_counts())
    nopatientdata = [patient for patient in list(dataset.index) if patient not in list(clin['Patient'])]
    dataset = dataset.drop(nopatientdata) # Remove patients with no metadata
    outputVector = getClinicalVector(list(dataset.index), clin, 'HR_FLAG')
    outputVector = np.array(outputVector) == 'TRUE'
    outputVector = outputVector.tolist()

    # datasetInfo(dataset, clin)
    if preprocessing:
        trainData = preprocess(dataset, method = method) # with preprocessing
    else:
        trainData = dataset.values

    print('Original dataset dimensions: ', dataset.shape)


    # Feature selection
    if featSel:
        # Feature selection
        newTrainData = featureSelection(trainData, outputVector, percentile = perc)
        print('Filtered dataset dimensions: ', newTrainData.shape)
    else:
        newTrainData = trainData

    models = ['knn', 'nbayes', 'decisionTree', 'logisticRegression', 'svm',
              'nnet', 'rand_forest', 'bagging']

    # Test models
    for model in models:
        modelTrain(newTrainData, outputVector, method = model)

    # Test models w/ parameter optimization
    # print('============= RNA SEQ DATA W/ PARAM OPT ============')
    # for model in models:
    #     if model is not 'nbayes':
    #         modelTrainOptimization(newTrainData, outputVector, method = model, print_res = True)


# print('\n =============== MICROARRAY DATA ================ \n')
# print('\n ========= SCALER PREPROCESSING ========= \n')
# testingMicroarrays(featSel = True, perc = 40, preprocessing = True, method = 'scaler')
# print('\n ========= LOGARITHM PREPROCESSING ========= \n')
# testingMicroarrays(featSel = True, perc = 40, preprocessing = True, method = 'log')
#
# print('\n ================= RNA SEQ DATA ================= \n')
# print('\n ========= SCALER PREPROCESSING ========= \n')
# testingRNASeq(featSel = True, perc = 40, preprocessing = True, method = 'scaler')
print('\n ========= LOGARITHM PREPROCESSING ========= \n')
#testingRNASeq(featSel = True, perc = 40, preprocessing = True, method = 'log')













