import pickle

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing.imputation import Imputer
from machinelearning.vcf_model_trainer import VCFModelTrainer
import os.path as path
from preprocessor.vcf_features_selector import VCFFeaturesSelector


def read_serialized_dataset(datasetpath):
    f = open(datasetpath, 'rb')
    dataset = pickle.load(f)
    return dataset

def serializeFeatures(modelsFolder, dataset, allX):
    columnsfilename = str(dataset.get_dataset_origin()) + "_featColumns_CH3.pkl"
    print("Saving " + columnsfilename)
    f = open(path.join(modelsFolder, columnsfilename), 'wb')
    pickle.dump(allX.columns, f)
    f.close()

def serializeSelectedFeatures(modelsFolder, dataset, features, featureGroupName):
    columnsfilename = str(dataset.get_dataset_origin()) +"_"+featureGroupName+ "_featColumns_CH3.pkl"
    print("Saving " + columnsfilename)
    f = open(path.join(modelsFolder, columnsfilename), 'wb')
    pickle.dump(features, f)
    f.close()

def generateTransformerName(modelsFolder, dataset, saveFiles):
    transformerfilename = None
    if saveFiles:
        transformerfilename = str(dataset.get_dataset_origin()) + "_Transformer_CH3.pkl"
        print("Saving " + transformerfilename)
        transformerfilename = path.join(modelsFolder, transformerfilename)
    return transformerfilename


def serializeClassifier(modelsFolder, dataset, clf):
    clffilename = str(dataset.get_dataset_origin()) + "_Classifier_CH3.pkl"
    print("Saving " + clffilename)
    f = open(path.join(modelsFolder, clffilename), 'wb')
    pickle.dump(clf,f)
    f.close()

def execute():
    
    modelsFolder = '/home/rrodrigues/Work'
    
    #datasetpath='/home/rrodrigues/Work/all-datasets/MuTectsnvs_filtered_dataset_CH1.pkl'
    #datasetpath='/home/rrodrigues/Work/all-datasets/Strelkasnvs_filtered_dataset_CH1.pkl'
    #datasetpath='/home/rrodrigues/Work/all-datasets/RNASeq_dataset_CH1.pkl'
    datasetpath='/home/rrodrigues/Work/all-datasets/MicroArrays_dataset_CH1.pkl'
    
    
    dataset = read_serialized_dataset(datasetpath)
    
    print(dataset.get_flags().value_counts())
    filterator = VCFFeaturesSelector(dataset)
    dataset = filterator.generateFilteredData()

    allX = dataset.getFullDataframe(False,False)
    serializeFeatures(modelsFolder, dataset, allX)

    trainer = VCFModelTrainer()
    inputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    #variance = VarianceThreshold(threshold=(.9 * (1 - .9)))
    variance = None
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=100)
    y = dataset.get_flags()
    print(len(y))
    X, y, z = trainer.df_reduce(allX, y, inputer, variance, scaler, fts, generateTransformerName(modelsFolder, dataset, True))
    columnsToCheck = allX.columns[z]
    print(columnsToCheck)
    trainer.doCrossValidation('nnet', X, y, folds=StratifiedKFold(n_splits=10, shuffle=False))
    clf = trainer.trainModel('nnet', X, y)
    serializeClassifier(modelsFolder, dataset, clf)

if __name__ == '__main__':
    execute()