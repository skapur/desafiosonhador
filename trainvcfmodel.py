import sys
sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')
import pickle

from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.preprocessing.data import StandardScaler

from machinelearning.vcf_model_trainer import VCFModelTrainer
import os.path as path
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor

def processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset):
    X = dataset.getFullDataframe(False, False)
    if saveFiles:
        columnsfilename = str(dataset.get_dataset_origin()) + "_featColumns_CH1.pkl"
        print("Saving " + columnsfilename)
        f = open(path.join(modelsFolder, columnsfilename), 'wb')
        pickle.dump(X.columns, f)
        f.close()
    scaler = StandardScaler()
    fts = SelectPercentile(percentile=fts_percentile)
    transformerfilename = None
    if saveFiles:
        transformerfilename = str(dataset.get_dataset_origin()) + "_Transformer_CH1.pkl"
        print("Saving " + transformerfilename)
        transformerfilename = path.join(modelsFolder, transformerfilename)
    y = dataset.get_flags()
    X, y, z = trainer.df_reduce(X, y, scaler, fts, transformerfilename)
    if doCV:
        print(dataset.get_dataset_origin())
        trainer.doCrossValidation(method, X, y)
    clf = trainer.trainModel(method, X, y)
    if saveFiles:
        clffilename = str(dataset.get_dataset_origin()) + "_Classifier_CH1.pkl"
        print("Saving " + clffilename)
        f = open(path.join(modelsFolder, clffilename), 'wb')
        pickle.dump(clf,f)
        f.close()

def train_serialize_models(clinicalfile, dataFolder='/test-data/', modelsFolder='/', fts_percentile=15, method="nnet", doCV=True, saveFiles=True, joinAllDatasets=False):
    preprocessor = VCFDataPreprocessor(clinicalfile)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, forTraining=True)
    trainer = VCFModelTrainer()
    if not joinAllDatasets:
        for dataset in datasets.values():
            processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset)
    else:
        dataset = preprocessor.joinDatasetsToSingleDataset(datasets)
        processDataset(modelsFolder, fts_percentile, method, doCV, saveFiles, trainer, dataset)

if __name__ == '__main__':
    clinicalfile = '/home/dreamchallenge/synapse/syn7222203/Clinical Data/globalClinTraining.csv'
    dataFolder = '/home/dreamchallenge/link-data/'
    modelsFolder = '/home/dreamchallenge/'
    train_serialize_models(clinicalfile, dataFolder, modelsFolder, doCV=True, saveFiles=True, joinAllDatasets=True)
