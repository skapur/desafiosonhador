from datastructures.patientdata import PatientData
import os.path as path
import pickle

serialized_Features_folder = 'serialized_features'

model_synonyms = {
    "MuTectRnaseq" : "MuTectsnvs",
    "MuTectRnaseq_filtered" : "MuTectsnvs_filtered",
    "StrelkaIndelsRnaseq" : "StrelkaIndels",
    "StrelkaIndelsRnaseq_filtered" : "StrelkaIndels_filtered",
    "StrelkasnvsRnaseq" : "Strelkasnvs",
    "StrelkasnvsRnaseq_filtered" : "Strelkasnvs_filtered"
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
        scoring = self.__generateFilteredGenesScoringDF(self.__data)
        if scoring is not None:
            filteredData.set_genes_scoring(scoring)
        
        functions = self.__generateFilteredGenesFunctionAssociatedDF(self.__data)
        if functions is not None:    
            filteredData.set_genes_function_associated(functions)
        tloads = self.__generateFilteredGenesTLODFeaturesDF(self.__data)
        if tloads is not None:
            filteredData.set_genes_tlod(tloads)
        cyto = self.__generateFilteredCytogeneticFeaturesDF(self.__data)
        if cyto is not None:
            filteredData.set_cytogenetic_features(cyto)
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
        dataframe = data.get_genes_scoring()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesScoring')
            self.__get_Column_Counts(features, dataframe, 'genesScoring', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesFunctionAssociatedDF(self, data):
        dataframe = data.get_genes_function_associated()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesFunctionAssociated')
            self.__get_Column_Counts(features, dataframe, 'genesFunctionAssociated', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredCytogeneticFeaturesDF(self, data):
        dataframe = data.get_cytogenetic_features()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('cytogeneticFeatures')
            self.__get_Column_Counts(features, dataframe, 'cytogeneticFeatures', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesTLODFeaturesDF(self, data):
        dataframe = data.get_genes_tlod()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesTlod')
            self.__get_Column_Counts(features, dataframe, 'genesTlod', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
        