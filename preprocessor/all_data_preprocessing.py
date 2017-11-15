from datastructures.patientdata import PatientData
from load_ch2_data import get_ch2_data
import numpy as np
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from preprocessor.vcf_features_selector import VCFFeaturesSelector


class AllDataPreprocessor(object):
    
    def __init__(self, submissionfile):
        if submissionfile is not None:
            self.__submissionfile = submissionfile
            self.__clinicalData = pd.read_csv(submissionfile)
            self.__clinicalData["Patient Index"] = self.__clinicalData.index
            self.__clinicalData.index = self.__clinicalData["Patient"]
    
    def getClinicalData(self):
        return self.__clinicalData        

    def __addVCFInformation(self, directoryFolder, useFiltered, forTraining, groupAges):
        vcfpreprocessor = VCFDataPreprocessor(self.__submissionfile)
        result = vcfpreprocessor.getPatientDataByDataset(directoryFolder, useFiltered, forTraining, groupAges)
        return result


    def __addExpressionInformation(self, directoryFolder, forTraining=False):
        result = {}
        rnaseq, rnaseqClinical, rnaseqFlags, microarrays, microarraysClinical, microarraysFlags = get_ch2_data(self.__submissionfile, directoryFolder, forTraining)
        
        if forTraining:
            valid_samples = rnaseqFlags != "CENSORED"
            rnaseq = rnaseq[valid_samples]
            rnaseqClinical = rnaseqClinical[valid_samples]
            rnaseqFlags = rnaseqFlags[valid_samples]
            rnaseqFlags = rnaseqFlags == 'TRUE'
            
            valid_samples2 = microarraysFlags!= "CENSORED"
            microarrays = microarrays[valid_samples2] 
            microarraysClinical = microarraysClinical[valid_samples2]
            microarraysFlags = microarraysFlags[valid_samples2]
            microarraysFlags = microarraysFlags == 'TRUE'

        if not rnaseq.empty:
            rnaseqData = PatientData("RNASeq", pd.Series(data=rnaseqClinical.index.values, name="Patient", index=rnaseqClinical.index.values))
            rnaseqData.set_ages(rnaseqClinical["D_Age"])
            rageRiskDF = rnaseqClinical["D_Age"].copy()
            rageRiskDF.name = "D_Age_Risk"
            rageRiskDF[rageRiskDF >= 65] = 1
            rageRiskDF[rageRiskDF < 65] = 0
            rnaseqData.set_ageRisk(rageRiskDF)
            rnaseqData.set_ISSs(rnaseqClinical["D_ISS"])
            if forTraining:
                rnaseqData.set_flags(rnaseqFlags)
            rnaseqData.set_genes_rnaseq(rnaseq)
            result["RNASeq"] = rnaseqData

        if not microarrays.empty:
            microarraysData = PatientData("MicroArrays", pd.Series(data=microarraysClinical.index.values, name="Patient", index=microarraysClinical.index.values))
            microarraysData.set_ages(microarraysClinical["D_Age"])
            mageRiskDF = microarraysClinical["D_Age"].copy()
            mageRiskDF.name = "D_Age_Risk"
            mageRiskDF[mageRiskDF >= 65] = 1
            mageRiskDF[mageRiskDF < 65] = 0
            microarraysData.set_ageRisk(mageRiskDF)
            microarraysData.set_ISSs(microarraysClinical["D_ISS"])
            if forTraining:
                microarraysData.set_flags(microarraysFlags)
            microarraysData.set_genes_microarray(microarrays)
            result["MicroArrays"] = microarraysData
            
        return result

    def getPatientDataByDataset(self, directoryFolder='/test-data/', useFiltered=True, forTraining=False, groupAges=False):
        x = self.__addVCFInformation(directoryFolder, useFiltered, forTraining, groupAges)
        y = self.__addExpressionInformation(directoryFolder, forTraining)
        result = {**x, **y}

        return result
    
    def prepareDatasetForStacking(self, datasets, useFiltering=True):
        if useFiltering:
            datasets = self.filterFeatureGroupsInDatasets(datasets)
        
        patients = None
        ages = None
        ageRisk = None
        iSSs = None
        genes_scoring = None
        genes_function_associated = None
        cytogenetic_features = None
        tlod = None
        qss = None
        big_qss = None
        clustered = None
        germline = None
        somaticrisk = None
        flags = None
        microarrays = None
        rnaseq = None
        containsFiltered = False
        for dataset in datasets.values():
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
                
            if ageRisk is None:
                ageRisk = dataset.get_ageRisk()
            else:
                ageRisk = pd.concat([ageRisk, dataset.get_ageRisk()])
                
            if iSSs is None:
                iSSs = dataset.get_ISSs()
            else:
                iSSs = pd.concat([iSSs, dataset.get_ISSs()])
            
            if cytogenetic_features is None:
                cytogenetic_features = dataset.get_cytogenetic_features()
            else:
                cytogenetic_features = pd.concat([cytogenetic_features, dataset.get_cytogenetic_features()])
                
            if flags is None:
                flags = dataset.get_flags()
            else:
                flags = pd.concat([flags, dataset.get_flags()])
                
            if genes_scoring is None:
                genes_scoring = self.__addDatasetNameToDataframe(dataset.get_genes_scoring(), dataset)
            else:
                genes_scoring = pd.concat([genes_scoring, self.__addDatasetNameToDataframe(dataset.get_genes_scoring(), dataset)], axis=1)
                
            if genes_function_associated is None:
                genes_function_associated = self.__addDatasetNameToDataframe(dataset.get_genes_function_associated(), dataset)
            else:
                genes_function_associated = pd.concat([genes_function_associated, self.__addDatasetNameToDataframe(dataset.get_genes_function_associated(), dataset)], axis=1)
                
            if tlod is None:
                tlod = self.__addDatasetNameToDataframe(dataset.get_genes_tlod(),dataset)
            elif dataset.get_genes_tlod() is not None:
                tlod = pd.concat([tlod, self.__addDatasetNameToDataframe(dataset.get_genes_tlod(),dataset)], axis=1)
                
            if qss is None:
                qss = dataset.get_genes_qss()
            elif dataset.get_genes_qss() is not None:
                qss = pd.concat([qss, self.__addDatasetNameToDataframe(dataset.get_genes_qss(),dataset)], axis=1)
                
            if big_qss is None:
                big_qss = dataset.get_genes_big_qss()
            elif  dataset.get_genes_big_qss() is not None:
                big_qss = pd.concat([big_qss, self.__addDatasetNameToDataframe(dataset.get_genes_big_qss(),dataset)], axis=1)
                
            if clustered is None:
                clustered = self.__addDatasetNameToDataframe(dataset.get_genes_clustered(),dataset)
            elif dataset.get_genes_clustered() is not None:
                clustered = pd.concat([clustered, self.__addDatasetNameToDataframe(dataset.get_genes_clustered(),dataset)], axis=1)
            
            if germline is None:
                germline = self.__addDatasetNameToDataframe(dataset.get_genes_germline_risk(), dataset)
            elif dataset.get_genes_germline_risk() is not None:
                germline = pd.concat([germline, self.__addDatasetNameToDataframe(dataset.get_genes_germline_risk(), dataset)], axis=1)
            
            if somaticrisk is None:
                somaticrisk = self.__addDatasetNameToDataframe(dataset.get_genes_somatic_risk(), dataset)
            elif dataset.get_genes_somatic_risk() is not None:
                somaticrisk = pd.concat([somaticrisk, self.__addDatasetNameToDataframe(dataset.get_genes_somatic_risk(), dataset)], axis=1)    
        
            if microarrays is None:
                microarrays = self.__addDatasetNameToDataframe(dataset.get_genes_microarray())
            elif dataset.get_genes_microarray() is not None:
                microarrays = pd.concat([microarrays, self.__addDatasetNameToDataframe(dataset.get_genes_microarray())])
            
            if rnaseq is None:
                rnaseq = self.__addDatasetNameToDataframe(dataset.get_genes_rnaseq())
            elif dataset.get_genes_rnaseq() is not None:
                rnaseq = pd.concat([rnaseq, self.__addDatasetNameToDataframe(dataset.get_genes_rnaseq())])
                
        data = None
        if patients is not None:
            patients = self.__processFirstOfGroupedDataFrame(patients)
            datasetname = 'ALL_Stacking'
            if containsFiltered:
                datasetname = datasetname + "_filtered"
            data = PatientData(datasetname, patients)
            if ages is not None:        
                data.set_ages(self.__processFirstOfGroupedDataFrame(ages))
            if ageRisk is not None:        
                data.set_ageRisk(self.__processFirstOfGroupedDataFrame(ageRisk))
            if iSSs is not None:
                data.set_ISSs(self.__processFirstOfGroupedDataFrame(iSSs))
            if genes_scoring is not None:
                data.set_genes_scoring(genes_scoring)
            if genes_function_associated is not None:
                data.set_genes_function_associated(genes_function_associated)
            if tlod is not None:
                data.set_genes_tlod(tlod)
            if qss is not None:
                data.set_genes_qss(qss)
            if big_qss is not None:
                data.set_genes_big_qss(big_qss)
            if clustered is not None:
                data.set_genes_clustered(clustered)
            if germline is not None:   
                data.set_genes_germline_risk(germline)
            if somaticrisk is not None:
                data.set_genes_somatic_risk(somaticrisk)
            if cytogenetic_features is not None:
                data.set_cytogenetic_features(self.__processFirstOfGroupedDataFrame(cytogenetic_features))
            if flags is not None:
                data.set_flags(self.__processFirstOfGroupedDataFrame(flags))
            if microarrays is not None:
                data.set_genes_microarray(self.__processFirstOfGroupedDataFrame(microarrays))
            if rnaseq is not None:
                data.set_genes_rnaseq(self.__processFirstOfGroupedDataFrame(rnaseq))
        return data

    def joinDatasetsToSingleDataset(self, datasets, useFiltering=True):
        if useFiltering:
            datasets = self.filterFeatureGroupsInDatasets(datasets)
        
        patients = None
        ages = None
        ageRisk = None
        iSSs = None
        genes_scoring = None
        genes_function_associated = None
        cytogenetic_features = None
        tlod = None
        qss = None
        big_qss = None
        clustered = None
        germline = None
        somaticrisk = None
        flags = None
        microarrays = None
        rnaseq = None
        containsFiltered = False
        for dataset in datasets.values():
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
                
            if ageRisk is None:
                ageRisk = dataset.get_ageRisk()
            else:
                ageRisk = pd.concat([ageRisk, dataset.get_ageRisk()])
                
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
                
            if tlod is None:
                tlod = dataset.get_genes_tlod()
            elif dataset.get_genes_tlod() is not None:
                tlod = pd.concat([tlod, dataset.get_genes_tlod()], axis=1)
                
            if qss is None:
                qss = dataset.get_genes_qss()
            elif dataset.get_genes_qss() is not None:
                qss = pd.concat([qss, dataset.get_genes_qss()], axis=1)
                
            if big_qss is None:
                big_qss = dataset.get_genes_big_qss()
            elif  dataset.get_genes_big_qss() is not None:
                big_qss = pd.concat([big_qss, dataset.get_genes_big_qss()], axis=1)
                
            if cytogenetic_features is None:
                cytogenetic_features = dataset.get_cytogenetic_features()
            else:
                cytogenetic_features = pd.concat([cytogenetic_features, dataset.get_cytogenetic_features()])
                
            if clustered is None:
                clustered = dataset.get_genes_clustered()
            elif dataset.get_genes_clustered() is not None:
                clustered = pd.concat([clustered, dataset.get_genes_clustered()], axis=1)
            
            if germline is None:
                germline = dataset.get_genes_germline_risk()
            elif dataset.get_genes_germline_risk() is not None:
                germline = pd.concat([germline, dataset.get_genes_germline_risk()], axis=1)
            
            if somaticrisk is None:
                somaticrisk = dataset.get_genes_somatic_risk()
            elif dataset.get_genes_somatic_risk() is not None:
                somaticrisk = pd.concat([somaticrisk, dataset.get_genes_somatic_risk()], axis=1)    
            
            if flags is None:
                flags = dataset.get_flags()
            else:
                flags = pd.concat([flags, dataset.get_flags()])
            
            if microarrays is None:
                microarrays = dataset.get_genes_microarray()
            elif dataset.get_genes_microarray() is not None:
                microarrays = pd.concat([microarrays, dataset.get_genes_microarray()])
            
            if rnaseq is None:
                rnaseq = dataset.get_genes_rnaseq()
            elif dataset.get_genes_rnaseq() is not None:
                rnaseq = pd.concat([rnaseq, dataset.get_genes_rnaseq()])
        
        data = None
        if patients is not None:
            patients = self.__processFirstOfGroupedDataFrame(patients)
            datasetname = 'ALL'
            if containsFiltered:
                datasetname = datasetname + "_filtered"
            data = PatientData(datasetname, patients)
            if ages is not None:        
                data.set_ages(self.__processFirstOfGroupedDataFrame(ages))
            if ageRisk is not None:        
                data.set_ageRisk(self.__processFirstOfGroupedDataFrame(ageRisk))
            if iSSs is not None:
                data.set_ISSs(self.__processFirstOfGroupedDataFrame(iSSs))
            if genes_scoring is not None:
                data.set_genes_scoring(self.__processBinaryGroupedDataFrame(genes_scoring))
            if genes_function_associated is not None:
                data.set_genes_function_associated(self.__processBinaryGroupedDataFrame(genes_function_associated))
            if tlod is not None:
                data.set_genes_tlod(self.__processBinaryGroupedDataFrame(tlod))
            if qss is not None:
                data.set_genes_qss(self.__processBinaryGroupedDataFrame(qss))
            if big_qss is not None:
                data.set_genes_big_qss(self.__processBinaryGroupedDataFrame(big_qss))
            if clustered is not None:
                data.set_genes_clustered(self.__processBinaryGroupedDataFrame(clustered))
            if germline is not None:   
                data.set_genes_germline_risk(self.__processBinaryGroupedDataFrame(germline))
            if somaticrisk is not None:
                data.set_genes_somatic_risk(self.__processBinaryGroupedDataFrame(somaticrisk))
            if cytogenetic_features is not None:
                data.set_cytogenetic_features(self.__processFirstOfGroupedDataFrame(cytogenetic_features))
            if flags is not None:
                data.set_flags(self.__processFirstOfGroupedDataFrame(flags))
            if microarrays is not None:
                data.set_genes_microarray(self.__processFirstOfGroupedDataFrame(microarrays))
            if rnaseq is not None:
                data.set_genes_rnaseq(self.__processFirstOfGroupedDataFrame(rnaseq))
        return data
    
    def filterFeatureGroupsInDatasets(self, datasets):
        for datasetkey in datasets.keys():
            dataset = datasets[datasetkey]
            selector = VCFFeaturesSelector(dataset)
            datasets[datasetkey] = selector.generateFilteredData()
        
        return datasets

    def __processFirstOfGroupedDataFrame(self, dataframe):
        dataframe = dataframe.groupby(dataframe.index).first()
        return dataframe
    
    def __processBinaryGroupedDataFrame(self, dataframe):
        dataframe = dataframe.groupby(dataframe.columns, axis=1).sum()
        dataframe[dataframe > 1] = 1
        dataframe[dataframe == 0] = np.nan
        dataframe = dataframe.dropna(axis=1, how='all')
        dataframe = dataframe.fillna(value=0)
        return dataframe
    
    def __addDatasetNameToDataframe(self, dataframe, dataset):
        if dataframe is not None:
            dataframe = dataframe.rename(columns = lambda x : dataset.get_dataset_origin()+ "_" + x)
        return dataframe