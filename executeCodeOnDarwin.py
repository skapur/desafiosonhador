import sys
import pickle
import os.path as path
from preprocessor.all_data_preprocessing import AllDataPreprocessor

sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')

def serializeDataset(modelsFolder, dataset):
    datasetfilename = str(dataset.get_dataset_origin()) + "_dataset_CH1.pkl"
    print("Saving " + datasetfilename)
    f = open(path.join(modelsFolder, datasetfilename), 'wb')
    pickle.dump(dataset, f)
    f.close()

def generate_datasets_forTraining(clinicalfile, dataFolder='/test-data/', datasetsFolder='/'):
    preprocessor = AllDataPreprocessor(clinicalfile)
    datasets = preprocessor.getPatientDataByDataset(dataFolder, useFiltered=True, forTraining=True)
    for dataset in datasets.values():
        serializeDataset(datasetsFolder, dataset)
    dataset = preprocessor.joinDatasetsToSingleDataset(datasets, False)
    serializeDataset(datasetsFolder, dataset)

if __name__ == '__main__':
    clinicalfile = '/home/dreamchallenge/synapse/syn7222203/Clinical Data/globalClinTraining.csv'
    dataFolder = '/home/dreamchallenge/link-data/'
    datasetsFolder = '/home/dreamchallenge/all-datasets'
    generate_datasets_forTraining(clinicalfile, dataFolder, datasetsFolder)