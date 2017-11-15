import pickle

from datastructures.patientdata import PatientData
import os.path as path


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
        filteredData.set_ageRisk(self.__data.get_ageRisk())
        filteredData.set_ISSs(self.__data.get_ISSs())
        filteredData.set_flags(self.__data.get_flags())
        if self.__data.get_genes_microarray() is not None:
            filteredData.set_genes_microarray(self.__data.get_genes_microarray())
        if self.__data.get_genes_rnaseq() is not None:
            filteredData.set_genes_rnaseq(self.__data.get_genes_rnaseq())
        scoring = self.__generateFilteredGenesScoringDF(self.__data)
        if scoring is not None:
            filteredData.set_genes_scoring(scoring)
        
        functions = self.__generateFilteredGenesFunctionAssociatedDF(self.__data)
        if functions is not None:    
            filteredData.set_genes_function_associated(functions)
        
        tloads = self.__generateFilteredGenesTLODFeaturesDF(self.__data)
        if tloads is not None:
            filteredData.set_genes_tlod(tloads)
        
        qss = self.__generateFilteredGenesQSSFeaturesDF(self.__data)
        if qss is not None:
            filteredData.set_genes_qss(qss)
        
        big_qss = self.__generateFilteredGenesBigQSSFeaturesDF(self.__data)
        if big_qss is not None:
            filteredData.set_genes_big_qss(big_qss)
        
        clustered = self.__generateFilteredGenesClusteredFeaturesDF(self.__data)
        if clustered is not None:
            filteredData.set_genes_clustered(clustered)   
                    
        germline = self.__generateFilteredGenesGermlineRiskQSSFeaturesDF(self.__data)
        if germline is not None:
            filteredData.set_genes_germline_risk(germline)   
                    
        somatic = self.__generateFilteredGenesSomaticRiskQSSFeaturesDF(self.__data)
        if somatic is not None:
            filteredData.set_genes_somatic_risk(somatic)   
               
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
    
    def __generateFilteredGenesQSSFeaturesDF(self, data):
        dataframe = data.get_genes_qss()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesQss')
            self.__get_Column_Counts(features, dataframe, 'genesQss', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesBigQSSFeaturesDF(self, data):
        dataframe = data.get_genes_big_qss()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesBigQss')
            self.__get_Column_Counts(features, dataframe, 'genesBigQss', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesClusteredFeaturesDF(self, data):
        dataframe = data.get_genes_clustered()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesClustered')
            self.__get_Column_Counts(features, dataframe, 'genesClustered', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesGermlineRiskQSSFeaturesDF(self, data):
        dataframe = data.get_genes_germline_risk()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesGermlineRisk')
            self.__get_Column_Counts(features, dataframe, 'genesGermlineRisk', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
    
    def __generateFilteredGenesSomaticRiskQSSFeaturesDF(self, data):
        dataframe = data.get_genes_somatic_risk()
        if dataframe is not None:
            features = self.__loadSerializedFeatures('genesSomaticRisk')
            self.__get_Column_Counts(features, dataframe, 'genesSomaticRisk', data.get_dataset_origin())
            filteredDataframe = dataframe.loc[:, features]
            return filteredDataframe
        return None
        