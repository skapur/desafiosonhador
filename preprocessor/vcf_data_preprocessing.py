import multiprocessing

from datastructures.patientdata import PatientData
import numpy as np
import os.path as path
import pandas as pd
from preprocessor.vcf_features_selector import VCFFeaturesSelector
from readers.vcfreader import VCFReader


GENOMIC_PROPS = {
    "MuTectsnvs" : "WES_mutationFileMutect",
    "StrelkaIndels" : "WES_mutationFileStrelkaIndel",
    "Strelkasnvs" : "WES_mutationFileStrelkaSNV",
    "MuTectRnaseq" : "RNASeq_mutationFileMutect",
    "StrelkaIndelsRnaseq" : "RNASeq_mutationFileStrelkaIndel",
    "StrelkasnvsRnaseq" : "RNASeq_mutationFileStrelkaSNV"
}

CYTOGENETICS_PROPS = ["CYTO_predicted_feature_01","CYTO_predicted_feature_02","CYTO_predicted_feature_03",
                      "CYTO_predicted_feature_04","CYTO_predicted_feature_05","CYTO_predicted_feature_06",
                      "CYTO_predicted_feature_07","CYTO_predicted_feature_08","CYTO_predicted_feature_09",
                      "CYTO_predicted_feature_10","CYTO_predicted_feature_11","CYTO_predicted_feature_12",
                      "CYTO_predicted_feature_13","CYTO_predicted_feature_14","CYTO_predicted_feature_15",
                      "CYTO_predicted_feature_16","CYTO_predicted_feature_17","CYTO_predicted_feature_18"
                      ]

class VCFDataPreprocessor(object):
    
    def __init__(self, submissionfile):
        if submissionfile is not None:
            self.__clinicalData = pd.read_csv(submissionfile)
            self.__clinicalData["Patient Index"] = self.__clinicalData.index
            self.__clinicalData.index = self.__clinicalData["Patient"]
        self.__executor = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    
    def getClinicalData(self):
        return self.__clinicalData;
    
    def getPatientDataByDataset(self, directoryFolder='/test-data/', useFiltered=False, forTraining=False, groupAges=False):
        result = {}
        reader = VCFReader()
        
        for dataset in GENOMIC_PROPS.keys():
            dataset_origin = dataset
            datasetDataframe = self.__clinicalData[self.__clinicalData[GENOMIC_PROPS[dataset]].notnull()].copy()
            if not useFiltered:
                datasetDataframe.replace({'.FILTERED': ''}, regex=True, inplace=True)
            else:
                dataset_origin = dataset_origin + "_filtered"
            if "HR_FLAG" in datasetDataframe.columns and forTraining:
                valid_samples = datasetDataframe["HR_FLAG"] != "CENSORED"
                datasetDataframe = datasetDataframe[valid_samples]
            if not datasetDataframe.empty:
                data = PatientData(dataset_origin, datasetDataframe.loc[datasetDataframe.index, "Patient"].copy())
                data = self.__fillClinicalData(data, datasetDataframe, forTraining, groupAges)
    
                filenames = datasetDataframe[GENOMIC_PROPS[dataset]].unique()
                paths = [ path.join(directoryFolder, f) for f in filenames]
                vcfgenescoredict = {}
                vcfgenesfunctiondict = {}
                vcfgenestloddict = {}
                vcfgenesqssdict = {}
                vcfgenesbigqssdict = {}
                vcfgenesclustereddict = {}
                vcfgenesgermlineriskdict = {}
                vcfgenessomaticriskdict = {}
                for k, v in zip(filenames, self.__executor.map(reader.readVCFFileFindCompression, paths)):
                    vcfgenescoredict[k] = v[0]
                    vcfgenesfunctiondict[k] = v[1]
                    vcfgenestloddict[k] = v[2]
                    vcfgenesqssdict[k] = v[3]
                    vcfgenesbigqssdict[k] = v[4]
                    vcfgenesclustereddict[k] = v[5]
                    vcfgenesgermlineriskdict[k] = v[6]
                    vcfgenessomaticriskdict[k] = v[7]
                    
                
                vcfGenesScoreDF = self.__tranfromVCFDictToVCFDataframe(vcfgenescoredict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesScoreDF[vcfGenesScoreDF < 500] = np.nan
                vcfGenesScoreDF = vcfGenesScoreDF.dropna(axis=1, how='all').fillna(value=0)
                vcfGenesScoreDF[vcfGenesScoreDF >= 500] = 1
                #vcfGenesScoreDF = vcfGenesScoreDF.fillna(value=0)
                data.set_genes_scoring(vcfGenesScoreDF)
                
                vcfGenesFunctDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesfunctiondict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesFunctDF = vcfGenesFunctDF.fillna(value=0)
                data.set_genes_function_associated(vcfGenesFunctDF)
                
                vcfGenesTLODDF = self.__tranfromVCFDictToVCFDataframe(vcfgenestloddict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesTLODDF = vcfGenesTLODDF.fillna(value=0)
                if not vcfGenesTLODDF.empty:
                    data.set_genes_tlod(vcfGenesTLODDF)
                    
                vcfGenesQSSDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesqssdict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesQSSDF = vcfGenesQSSDF.fillna(value=0)
                if not vcfGenesQSSDF.empty:
                    data.set_genes_qss(vcfGenesQSSDF)
                
                vcfGenesBigQSSDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesbigqssdict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesBigQSSDF = vcfGenesBigQSSDF.fillna(value=0)
                if not vcfGenesBigQSSDF.empty:
                    data.set_genes_big_qss(vcfGenesBigQSSDF)    
                
                data.set_cytogenetic_features(datasetDataframe[CYTOGENETICS_PROPS])

                vcfGenesclusteredDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesclustereddict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesclusteredDF = vcfGenesclusteredDF.fillna(value=0)
                if not vcfGenesclusteredDF.empty:
                    data.set_genes_clustered(vcfGenesclusteredDF)
                    
                vcfGenesGermlineRiskDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesgermlineriskdict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesGermlineRiskDF = vcfGenesGermlineRiskDF.fillna(value=0)
                if not vcfGenesGermlineRiskDF.empty:
                    data.set_genes_germline_risk(vcfGenesGermlineRiskDF)
                    
                vcfGenesSomaticRiskDF = self.__tranfromVCFDictToVCFDataframe(vcfgenessomaticriskdict, datasetDataframe, GENOMIC_PROPS[dataset])
                vcfGenesSomaticRiskDF = vcfGenesSomaticRiskDF.fillna(value=0)
                if not vcfGenesSomaticRiskDF.empty:
                    data.set_genes_somatic_risk(vcfGenesSomaticRiskDF)
                
                if not vcfGenesScoreDF.empty:
                    result[dataset_origin] = data

        return result
    

    def filterFeatureGroupsInDatasets(self, datasets):
        for datasetkey in datasets.keys():
            dataset = datasets[datasetkey]
            selector = VCFFeaturesSelector(dataset)
            datasets[datasetkey] = selector.generateFilteredData()
        
        return datasets
    
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
        return data
    
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
     
    def __fillClinicalData(self, patientdata, datasetDataframe, forTraining=False, groupAges=False):
        colums = self.__clinicalData.columns
        containsAge = "D_Age" in colums
        containsISS = "D_ISS" in colums
        containsHRFlag = "HR_FLAG" in colums
        
        if containsAge:
            ageDF = datasetDataframe.loc[datasetDataframe.index, "D_Age"].copy()
            ageRiskDF = datasetDataframe.loc[datasetDataframe.index, "D_Age"].copy()
            if groupAges:
                for row in ageDF.index:
                    ageDF.at[row] = self.__ageToGroup(ageDF[row])
            patientdata.set_ages(ageDF)
            ageRiskDF.name = "D_Age_Risk"
            ageRiskDF[ageRiskDF >= 65] = 1
            ageRiskDF[ageRiskDF < 65] = 0
            patientdata.set_ageRisk(ageRiskDF)
            
        if containsISS:
            patientdata.set_ISSs(datasetDataframe.loc[datasetDataframe.index, "D_ISS"].copy())
            
        if containsHRFlag and forTraining:
            flags = datasetDataframe.loc[datasetDataframe.index, "HR_FLAG"].copy()
            flags = flags == 'TRUE'
            patientdata.set_flags(flags)
            
        return patientdata
               
    def __tranfromVCFDictToVCFDataframe(self, vcfDict, datasetDataframe, datasetcolumn):
        vcfDataframe = pd.DataFrame(vcfDict)
        vcfDataframe = vcfDataframe.T
        vcfDataframe.fillna(value=0, inplace=True)
        datasetDataframe = datasetDataframe.loc[datasetDataframe.index, ["Patient", datasetcolumn]]
        patientbyfilename = datasetDataframe.set_index(datasetcolumn, drop=True, append=False, inplace=False)
        patientbyfilename = patientbyfilename.join(vcfDataframe)
        patientbyfilename = patientbyfilename.loc[pd.notnull(patientbyfilename.index)]
        vcfDataframe = patientbyfilename.set_index("Patient", drop=True, append=False, inplace=False)
        return vcfDataframe
        
    def __ageToGroup(self, age):
        if age is np.nan:
            return np.nan;
        if age < 18:
            return 1;
        if age >=18 and age <30:
            return 2;
        if age >=30 and age <40:
            return 3;
        if age >= 40 and age <50:
            return 4
        if age >= 50 and age <65:
            return 5;
        if age >=65:
            return 6;
    def __addDatasetNameToDataframe(self, dataframe, dataset):
        if dataframe is not None:
            dataframe = dataframe.rename(columns = lambda x : dataset.get_dataset_origin()+ "_" + x)
        return dataframe
        