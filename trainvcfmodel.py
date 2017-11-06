from builtins import range
from copyreg import pickle
import pickle
import sys

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing.imputation import Imputer

from datastructures.patientdata import PatientData
from machinelearning.vcf_model_predictor import VCFModelPredictor
from machinelearning.vcf_model_trainer import VCFModelTrainer
import numpy as np
import os.path as path
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from preprocessor.vcf_features_selector import VCFFeaturesSelector


sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')

def serializeDataset(modelsFolder, dataset):
    datasetfilename = str(dataset.get_dataset_origin()) + "_dataset_CH1.pkl"
    print("Saving " + datasetfilename)
    f = open(path.join(modelsFolder, datasetfilename), 'wb')
    pickle.dump(dataset, f)
    f.close()

def serializeFeatures(modelsFolder, dataset, allX):
    columnsfilename = str(dataset.get_dataset_origin()) + "_featColumns_CH1.pkl"
    print("Saving " + columnsfilename)
    f = open(path.join(modelsFolder, columnsfilename), 'wb')
    pickle.dump(allX.columns, f)
    f.close()

def serializeSelectedFeatures(modelsFolder, dataset, features, featureGroupName):
    columnsfilename = str(dataset.get_dataset_origin()) +"_"+featureGroupName+ "_featColumns_CH1.pkl"
    print("Saving " + columnsfilename)
    f = open(path.join(modelsFolder, columnsfilename), 'wb')
    pickle.dump(features, f)
    f.close()

def generateTransformerName(modelsFolder, dataset, saveFiles):
    transformerfilename = None
    if saveFiles:
        transformerfilename = str(dataset.get_dataset_origin()) + "_Transformer_CH1.pkl"
        print("Saving " + transformerfilename)
        transformerfilename = path.join(modelsFolder, transformerfilename)
    return transformerfilename

def serializeClassifier(modelsFolder, dataset, clf):
    clffilename = str(dataset.get_dataset_origin()) + "_Classifier_CH1.pkl"
    print("Saving " + clffilename)
    f = open(path.join(modelsFolder, clffilename), 'wb')
    pickle.dump(clf,f)
    f.close()

def processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset, savedataset=False):
    #filterator = VCFFeaturesSelector(dataset)
    #dataset = filterator.generateFilteredData()
    allX = dataset.getFullDataframe(False, False)
    if allX is not None:
        if savedataset:
            serializeDataset(modelsFolder, dataset)
        if saveFiles:
            serializeFeatures(modelsFolder, dataset, allX)
        variance = None
        scaler = StandardScaler()
        fts = SelectPercentile(percentile=fts_percentile)
        y = dataset.get_flags()
        X, y, z = trainer.df_reduce(allX, y, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, saveFiles))
        print(allX.columns[z])
        if doCV:
            print(dataset.get_dataset_origin())
            trainer.doCrossValidation(method, X, y)
        clf = trainer.trainModel(method, X, y)
        if saveFiles:
            serializeClassifier(modelsFolder, dataset, clf)

def generate_datasets_forTraining(clinicalfile, dataFolder='/test-data/', datasetsFolder='/'):
    preprocessor = VCFDataPreprocessor(clinicalfile)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, useFiltered=False, forTraining=True)
    for dataset in datasets.values():
        serializeDataset(datasetsFolder, dataset)
    dataset = preprocessor.joinDatasetsToSingleDataset(datasets, False)
    serializeDataset(datasetsFolder, dataset)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, useFiltered=True, forTraining=True)
    for dataset in datasets.values():
        serializeDataset(datasetsFolder, dataset)
    dataset = preprocessor.joinDatasetsToSingleDataset(datasets, False)
    serializeDataset(datasetsFolder, dataset)

def train_serialize_models(clinicalfile, dataFolder='/test-data/', modelsFolder='/', fts_percentile=10, method="nnet", doCV=True, saveFiles=True, joinAllDatasets=False, savedataset=False):
    preprocessor = VCFDataPreprocessor(clinicalfile)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, useFiltered=False, forTraining=True)
    trainer = VCFModelTrainer()
    if not joinAllDatasets:
        for dataset in datasets.values():
            processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset, savedataset)
    else:
        dataset = preprocessor.joinDatasetsToSingleDataset(datasets)
        processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset, savedataset)

def read_serialized_dataset(datasetpath):
    f = open(datasetpath, 'rb')
    dataset = pickle.load(f)
    return dataset

def executeCodeOnDarwin():
    clinicalfile = '/home/dreamchallenge/synapse/syn7222203/Clinical Data/globalClinTraining.csv'
    dataFolder = '/home/dreamchallenge/link-data/'
    datasetsFolder = '/home/dreamchallenge/vcf-datasets_v3'
    #train_serialize_models(clinicalfile, dataFolder, modelsFolder, doCV=False, saveFiles=False, fts_percentile=2, joinAllDatasets=False, savedataset=True)
    generate_datasets_forTraining(clinicalfile, dataFolder, datasetsFolder)
    
def executeCodeManually():

    modelsFolder = '/home/tiagoalves/rrodrigues/'
    datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v4/MuTectsnvs_filtered_dataset_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v4/StrelkaIndels_dataset_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v4/Strelkasnvs_filtered_dataset_CH1.pkl'
    
    dataset = read_serialized_dataset(datasetpath)
    
    print(dataset.get_flags().value_counts())
    print(dataset.getFullDataframe(False,False).columns)
    filterator = VCFFeaturesSelector(dataset)
    dataset = filterator.generateFilteredData()
    allX = dataset.getFullDataframe(False,False)
    
    #allX = dataset.get_genes_scoring() #5.9
    #allX = dataset.get_genes_function_associated()
    #allX = dataset.get_cytogenetic_features()
    #allX = dataset.get_genes_tlod()
    #allX = dataset.get_genes_qss()
    #allX = dataset.get_genes_big_qss()
    #allX = dataset.getFullDataframe(False,False)
    
    serializeFeatures(modelsFolder, dataset, allX)
    
    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    #variance = VarianceThreshold(threshold=(.9 * (1 - .9)))
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=100)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, True))
    print(allX.columns[z])
    #serializeSelectedFeatures(modelsFolder, dataset, allX.columns[z], "genesBigQss")
    #trainer.testAllMethodsCrossValidation(X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    trainer.doCrossValidation('logisticRegression', X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    clf = trainer.trainModel('logisticRegression', X, y)
    serializeClassifier(modelsFolder, dataset, clf)
    
def executeJoinModelCodeManually():
    modelsFolder = '/home/tiagoalves/rrodrigues/'
    paths = ['/home/tiagoalves/rrodrigues/vcf-datasets_v4/MuTectsnvs_filtered_dataset_CH1.pkl',
             #'/home/tiagoalves/rrodrigues/vcf-datasets/StrelkaIndels_dataset_CH1.pkl',
             '/home/tiagoalves/rrodrigues/vcf-datasets_v4/Strelkasnvs_filtered_dataset_CH1.pkl']
    
    datasets = {}
    preprocessor = VCFDataPreprocessor(None)
    for pat in paths:
        dataset = read_serialized_dataset(pat)
        datasets[dataset.get_dataset_origin()] = dataset
    
    dataset = preprocessor.joinDatasetsToSingleDataset(datasets)
    allX = dataset.getFullDataframe(False,False)
    
    serializeFeatures(modelsFolder, dataset, allX)
    
    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=71)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, True))
    columnsToCheck = allX.columns[z]
    print(columnsToCheck)
    checkifinDataset(columnsToCheck, dataset.get_genes_scoring().columns, "Genes Scoring")
    checkifinDataset(columnsToCheck, dataset.get_genes_function_associated().columns, "Genes Function")
    checkifinDataset(columnsToCheck, dataset.get_genes_tlod().columns, "Genes Tlod")
    checkifinDataset(columnsToCheck, dataset.get_genes_qss().columns, "Genes QSS")
    checkifinDataset(columnsToCheck, dataset.get_genes_big_qss().columns, "Genes Big Qss")
    compareNames(dataset.get_genes_qss().columns, dataset.get_genes_big_qss().columns)
    #trainer.testAllMethodsCrossValidation(X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    trainer.doCrossValidation('logisticRegression', X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    clf = trainer.trainModel('logisticRegression', X, y)
    serializeClassifier(modelsFolder, dataset, clf)

def compareNames(columns1, columns2):
    columns1 = [x.replace("QSS_", "") for x in columns1]
    columns1 = set(columns1)
    #columns2 = [x.replace("TLOD_","") for x in columns2]
    #columns2 = [x.replace("QSS_", "") for x in columns2]
    columns2 = [x.replace("BIGQSS_", "") for x in columns2]
    columns2 = set(columns2)
    inter = columns1 & columns2
    print("names")
    print("inter size: "+str(len(inter)))
    print("percentage in genes scoring: " + str(len(inter)/len(columns1)))

def checkifinDataset(columnsToCheck, allColumns, datasetname):
    columnsToCheck = set(columnsToCheck)
    allColumns = set(allColumns)
    intersection = allColumns & columnsToCheck
    print(datasetname)
    intersize = len(intersection)
    print("inter size: "+str(intersize))
    print("percentage in all: " + str(intersize/len(allColumns)))

def compareUnfilteredVSFiltered():
    unfilteredpaths = ['/home/tiagoalves/rrodrigues/loaded-datasets/MuTectsnvs_dataset_Unfiltered_CH1.pkl',
                       
                       '/home/tiagoalves/rrodrigues/loaded-datasets/Strelkasnvs_dataset_Unfiltered_CH1.pkl']
    #'/home/tiagoalves/rrodrigues/loaded-datasets/StrelkaIndels_dataset_Unfiltered_CH1.pkl',
    
    filteredpaths = ['/home/tiagoalves/rrodrigues/loaded-datasets/MuTectsnvs_dataset_filtered_CH1.pkl',
                       
                       '/home/tiagoalves/rrodrigues/loaded-datasets/Strelkasnvs_dataset_filtered_CH1.pkl']
    #'/home/tiagoalves/rrodrigues/loaded-datasets/StrelkaIndels_dataset_filtered_CH1.pkl',
    
    unfiltereddatasets = [read_serialized_dataset(datasetpath) for datasetpath in unfilteredpaths]
    filtereddatasets = [read_serialized_dataset(datasetpath) for datasetpath in filteredpaths]
    
    for x in range(0, len(unfiltereddatasets),2):
        filtered = filtereddatasets[x].get_genes_function_associated()
        unfiltered = unfiltereddatasets[x].get_genes_function_associated()[filtered.columns]
        sum = filtered + unfiltered
        print(sum.shape)
        nonfiltereduniques = sum.loc[sum[sum.columns] == 1]
        print(nonfiltereduniques.shape)

    '''
    print("unfiltered")
    for unfiltereddataset in unfiltereddatasets:
        print(unfiltereddataset.get_dataset_origin())
        evaluateDatasetModel(unfiltereddataset)
    
    print("filtered")
    for filtereddataset in filtereddatasets:
        print(filtereddataset.get_dataset_origin())
        evaluateDatasetModel(filtereddataset)
    '''

def evaluateDatasetModel(dataset):
    if dataset.get_genes_scoring() is not None and not dataset.get_genes_scoring().empty: 
        selector = VCFFeaturesSelector(dataset)
        dataset = selector.generateFilteredData()
        X = dataset.getFullDataframe(False,False)
        predictor = VCFModelPredictor()
        true_y = dataset.get_flags().tolist()
        pred_y, scores = predictor.generate_predictions_scores(X, dataset.get_dataset_origin())
        print(classification_report(true_y, pred_y))
    

def checkFeaturePercentage():
    columns='/home/tiagoalves/rrodrigues/desafiosonhador/serialized_features/Strelkasnvs_filtered_genesQss_featColumns_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v3/MuTectsnvs_filtered_dataset_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v3/StrelkaIndels_dataset_CH1.pkl'
    datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets_v4/Strelkasnvs_filtered_dataset_CH1.pkl'
    
    dataset = read_serialized_dataset(datasetpath)
    print(dataset.getFullDataframe(False, False).columns)
    #allX = dataset.get_genes_scoring() #5.9
    #allX = dataset.get_genes_function_associated()
    #allX = dataset.get_cytogenetic_features()
    #allX = dataset.get_genes_tlod()
    allX = dataset.get_genes_qss()
    #allX = dataset.get_genes_big_qss()
    #filterator = VCFFeaturesSelector(dataset)
    #dataset = filterator.generateFilteredData()
    #allX = dataset.getFullDataframe(False,False)
    f = open(columns, 'rb')
    selectedcolumns = set(pickle.load(f))
    f.close()
    print(len(selectedcolumns) /len(allX.columns))
    
    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=6.1)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, None)
    result = set(allX.columns[z])
    print(len(selectedcolumns)/len(result))
    print(len(result))
    print(len(selectedcolumns))
    print(selectedcolumns-result)
    
def checkmodel():
    f = open("/home/tiagoalves/rrodrigues/desafiosonhador/serialized_models/MuTectsnvs_filtered_Classifier_CH1.pkl", "rb")
    clf = pickle.load(f)
    f.close()
    print(clf.__class__.__name__)

if __name__ == '__main__':
    #executeCodeOnDarwin()
    executeCodeManually()
    #executeJoinModelCodeManually()
    #compareUnfilteredVSFiltered()
    #checkFeaturePercentage()
    #checkmodel()
