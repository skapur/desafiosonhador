from datastructures.patientdata import PatientData
import os.path as path
import pickle

serialized_Features_folder = 'serialized_features'

model_synonyms = {
    "MuTectRnaseq" : "MuTectsnvs",
    "StrelkaIndelsRnaseq" : "StrelkaIndels",
    "StrelkasnvsRnaseq" : "Strelkasnvs"
}

class VCFFeaturesSelector(object):
    
    def __init__(self, data):

        self.__data = data
        if data.get_dataset_origin() in model_synonyms.keys():
            self.__modelType = model_synonyms[data.get_dataset_origin()]
        else:
            self.__modelType = data.get_dataset_origin()
    
    def generateFilteredData(self):
        filteredData = PatientData(self.__data.get_dataset_origin(), self.__data.get_patients())
        filteredData.set_ages(self.__data.get_ages())
        filteredData.set_ISSs(self.__data.get_ISSs())
        filteredData.set_flags(self.__data.get_flags())
        filteredData.set_genes_scoring(self.__generateFilteredGenesScoringDF(self.__data))
        filteredData.set_genes_function_associated(self.__generateFilteredGenesFunctionAssociatedDF(self.__data))
        filteredData.set_cytogenetic_features(self.__generateFilteredCytogeneticFeaturesDF(self.__data))
        return filteredData
    
    def __loadSerializedFeatures(self, featureGroupName):
        featuresFilename = self.__modelType + "_" + featureGroupName + "_featColumns_CH1.pkl"
        print("Loaded: " + featuresFilename)
        f = open(path.join(serialized_Features_folder, featuresFilename), 'rb')
        features = pickle.load(f)
        f.close()
        return features
    
    def __get_Column_Counts(self, features, dataframe, featrueGroupName, datasetOrigin):
        trues = 0.0
        falses = 0.0
        for feature in features:
            if feature in dataframe.columns:
                trues = trues + 1.0
            else:
                falses = falses + 1.0
        print("="*40)
        print("Dataset origin: " + datasetOrigin)
        print("Feature Group Name: " + featrueGroupName)
        print("Percentage of overlap: " + str(trues/len(features)*100)+"%")
        print("Num lost features: " + str(falses))
        print("Num generated columns: " + str(len(dataframe.columns)))
        print("num selected columns: " + str(trues))
        print("="*40)
        
    def __generateFilteredGenesScoringDF(self, data):
        features = self.__loadSerializedFeatures('genesScoring')
        dataframe = data.get_genes_scoring()
        self.__get_Column_Counts(features, dataframe, 'genesScoring', data.get_dataset_origin())
        filteredDataframe = dataframe.loc[:, features]
        filteredDataframe = filteredDataframe.fillna(value=0)
        return filteredDataframe
    
    def __generateFilteredGenesFunctionAssociatedDF(self, data):
        features = self.__loadSerializedFeatures('genesFunctionAssociated')
        dataframe = data.get_genes_function_associated()
        self.__get_Column_Counts(features, dataframe, 'genesFunctionAssociated', data.get_dataset_origin())
        filteredDataframe = dataframe.loc[:, features]
        filteredDataframe = filteredDataframe.fillna(value=0)
        return filteredDataframe
    
    def __generateFilteredCytogeneticFeaturesDF(self, data):
        features = self.__loadSerializedFeatures('cytogeneticFeatures')
        dataframe = data.get_cytogenetic_features()
        self.__get_Column_Counts(features, dataframe, 'cytogeneticFeatures', data.get_dataset_origin())
        filteredDataframe = dataframe.loc[:, features]
        filteredDataframe = filteredDataframe.fillna(value=-1)
        return filteredDataframe
        