from builtins import range
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
import os.path as path
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from preprocessor.vcf_features_selector import VCFFeaturesSelector


sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')

def joinDatasetsToSingleDataset(datasets):
    
    patients = None
    ages = None
    iSSs = None
    genes_scoring = None
    genes_function_associated = None
    cytogenetic_features = None
    flags = None
    containsFiltered = False
    
    for dataset in datasets:
        if "filtered" in dataset.get_dataset_origin():
            containsFiltered = True
        if patients is None: 
            patients = dataset.get_patients()
        else:
            patients = pd.concat([patients, dataset.get_patients()])
        if ages is None:
            ages = dataset.get_ages()
        else:
            ages = pd.concat([ages, dataset.get_ages()])
        if iSSs is None:
            iSSs = dataset.get_ISSs()
        else:
            iSSs = pd.concat([iSSs, dataset.get_ISSs()])
        if genes_scoring is None:
            genes_scoring = dataset.get_genes_scoring()
        else:
            genes_scoring = pd.concat([genes_scoring, dataset.get_genes_scoring()], axis=1)
        if genes_function_associated is None:
            genes_function_associated = dataset.get_genes_function_associated()
        else:
            genes_function_associated = pd.concat([genes_function_associated, dataset.get_genes_function_associated()], axis=1)
        if cytogenetic_features is None:
            cytogenetic_features = dataset.get_cytogenetic_features()
        else:
            cytogenetic_features = pd.concat([cytogenetic_features, dataset.get_cytogenetic_features()])
        if flags is None:
            flags = dataset.get_flags()
        else:
            flags = pd.concat([flags, dataset.get_flags()])
    
    data = None
    if patients is not None:
        patients = patients.groupby(patients.index).first()
        datasetname = 'ALL'
        if containsFiltered:
            datasetname = datasetname + "_filtered"
        data = PatientData(datasetname, patients)
        if ages is not None:        
            ages = ages.groupby(ages.index).first()
            data.set_ages(ages)
        if iSSs is not None:
            iSSs = iSSs.groupby(iSSs.index).first()
            data.set_ISSs(iSSs)
        if genes_scoring is not None:
            genes_scoring = genes_scoring.groupby(genes_scoring.columns, axis=1).sum()
            genes_scoring = genes_scoring.fillna(value=0)
            data.set_genes_scoring(genes_scoring)
        if genes_function_associated is not None:
            genes_function_associated = genes_function_associated.groupby(genes_function_associated.columns, axis=1).sum()
            genes_function_associated[genes_function_associated > 1] = 1
            genes_function_associated = genes_function_associated.fillna(value=0)
            data.set_genes_function_associated(genes_function_associated)
        if cytogenetic_features is not None:
            cytogenetic_features = cytogenetic_features.groupby(cytogenetic_features.index).first()
            data.set_cytogenetic_features(cytogenetic_features)
        if flags is not None:
            flags = flags.groupby(flags.index).first()
            data.set_flags(flags)
    return data

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
    dataset = joinDatasetsToSingleDataset(datasets.values())
    serializeDataset(datasetsFolder, dataset)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, useFiltered=True, forTraining=True)
    for dataset in datasets.values():
        serializeDataset(datasetsFolder, dataset)
    dataset = joinDatasetsToSingleDataset(datasets.values())
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
    datasetsFolder = '/home/dreamchallenge/vcf-datasets'
    #train_serialize_models(clinicalfile, dataFolder, modelsFolder, doCV=False, saveFiles=False, fts_percentile=2, joinAllDatasets=False, savedataset=True)
    generate_datasets_forTraining(clinicalfile, dataFolder, datasetsFolder)
    
def executeCodeManually():

    modelsFolder = '/home/tiagoalves/rrodrigues/'
    datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets/MuTectsnvs_filtered_dataset_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets/StrelkaIndels_dataset_CH1.pkl'
    #datasetpath='/home/tiagoalves/rrodrigues/vcf-datasets/Strelkasnvs_filtered_dataset_CH1.pkl'
    
    dataset = read_serialized_dataset(datasetpath)
    
    print(dataset.get_flags().value_counts())
    filterator = VCFFeaturesSelector(dataset)
    dataset = filterator.generateFilteredData()
    allX = dataset.getFullDataframe(False,False)
    
    #allX = dataset.get_genes_scoring()
    #allX = dataset.get_genes_function_associated()
    #allX = dataset.get_cytogenetic_features()
    #allX = dataset.getFullDataframe(False,False)
    
    serializeFeatures(modelsFolder, dataset, allX)
    
    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    #variance = VarianceThreshold(threshold=(.9 * (1 - .9)))
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=99)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, True))
    print(allX.columns[z])

    #serializeSelectedFeatures(modelsFolder, dataset, allX.columns[z], "cytogeneticFeatures")
    #trainer.testAllMethodsCrossValidation(X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    trainer.doCrossValidation('logisticRegression', X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    clf = trainer.trainModel('logisticRegression', X, y)
    serializeClassifier(modelsFolder, dataset, clf)
    
def executeJoinModelCodeManually():
    modelsFolder = '/home/tiagoalves/rrodrigues/'
    paths = ['/home/tiagoalves/rrodrigues/vcf-datasets/MuTectsnvs_filtered_dataset_CH1.pkl',
             #'/home/tiagoalves/rrodrigues/vcf-datasets/StrelkaIndels_dataset_CH1.pkl',
             '/home/tiagoalves/rrodrigues/vcf-datasets/Strelkasnvs_filtered_dataset_CH1.pkl']
    
    datasets = []
    
    for pat in paths:
        dataset = read_serialized_dataset(pat)
        filterator = VCFFeaturesSelector(dataset)
        dataset = filterator.generateFilteredData()
        datasets.append(dataset)
    
    dataset = joinDatasetsToSingleDataset(datasets)
    allX = dataset.getFullDataframe(False,False)
    
    serializeFeatures(modelsFolder, dataset, allX)
    
    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=69)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, True))
    print(allX.columns[z])
    
    trainer.doCrossValidation('logisticRegression', X, y, folds=StratifiedKFold(n_splits=10, shuffle=True))
    clf = trainer.trainModel('logisticRegression', X, y)
    serializeClassifier(modelsFolder, dataset, clf)

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
    

if __name__ == '__main__':
    #executeCodeOnDarwin()
    #executeCodeManually()
    executeJoinModelCodeManually()
    #compareUnfilteredVSFiltered()
