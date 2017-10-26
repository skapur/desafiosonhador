import multiprocessing
import os.path as path
import pandas as pd
import numpy as np
from readers.vcfreader import VCFReader
from datastructures.patientdata import PatientData
from preprocessor.vcf_features_selector import VCFFeaturesSelector



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
                for k, v in zip(filenames, self.__executor.map(reader.readVCFFileFindCompression, paths)):
                    vcfgenescoredict[k] = v[0]
                    vcfgenesfunctiondict[k] = v[1]
                    vcfgenestloddict[k] = v[2]
                
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
                
                data.set_cytogenetic_features(datasetDataframe[CYTOGENETICS_PROPS])
                
                if not vcfGenesScoreDF.empty:
                    result[dataset_origin] = data

        return result
    

    def filterFeatureGroupsInDatasets(self, datasets):
        for datasetkey in datasets.keys():
            dataset = datasets[datasetkey]
            selector = VCFFeaturesSelector(dataset)
            datasets[datasetkey] = selector.generateFilteredData()
        
        return datasets

    def joinDatasetsToSingleDataset(self, datasets, useFiltering=True):
        if useFiltering:
            datasets = self.filterFeatureGroupsInDatasets(datasets)
        
        patients = None
        ages = None
        iSSs = None
        genes_scoring = None
        genes_function_associated = None
        cytogenetic_features = None
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
                genes_scoring[genes_scoring > 1] = 1
                genes_scoring[genes_scoring == 0] = np.nan
                genes_scoring = genes_scoring.dropna(axis=1, how='all')
                #genes_scoring = genes_scoring.groupby(genes_scoring.columns, axis=1).sum() 
                genes_scoring = genes_scoring.fillna(value=0)
                data.set_genes_scoring(genes_scoring)
            if genes_function_associated is not None:
                genes_function_associated = genes_function_associated.groupby(genes_function_associated.columns, axis=1).sum()
                genes_function_associated[genes_function_associated > 1] = 1
                genes_function_associated[genes_function_associated == 0] = np.nan
                genes_function_associated = genes_function_associated.dropna(axis=1, how='all')
                genes_function_associated = genes_function_associated.fillna(value=0)
                data.set_genes_function_associated(genes_function_associated)
            if cytogenetic_features is not None:
                cytogenetic_features = cytogenetic_features.groupby(cytogenetic_features.index).first()
                data.set_cytogenetic_features(cytogenetic_features)
            if flags is not None:
                flags = flags.groupby(flags.index).first()
                data.set_flags(flags)
        return data
     
    def __fillClinicalData(self, patientdata, datasetDataframe, forTraining=False, groupAges=False):
        colums = self.__clinicalData.columns
        containsAge = "D_Age" in colums
        containsISS = "D_ISS" in colums
        containsHRFlag = "HR_FLAG" in colums
        
        if containsAge:
            ageDF = datasetDataframe.loc[datasetDataframe.index, "D_Age"].copy()
            ageRiskDF = ageDF.copy()
            if groupAges:
                for row in ageDF.index:
                    ageDF.at[row] = self.__ageToGroup(ageDF[row])
            patientdata.set_ages(ageDF)
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
        