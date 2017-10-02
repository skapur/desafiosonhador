import multiprocessing
import os.path as path
import pandas as pd
import numpy as np
from readers.vcfreader import VCFReader
from datastructures.patientdata import PatientData



GENOMIC_PROPS = {
    "MuTectsnvs" : "WES_mutationFileMutect",
    "StrelkaIndels" : "WES_mutationFileStrelkaIndel",
    "Strelkasnvs" : "WES_mutationFileStrelkaSNV",
    "MuTectRnaseq" : "RNASeq_mutationFileMutect",
    "StrelkaIndelsRnaseq" : "RNASeq_mutationFileStrelkaIndel",
    "StrelkasnvsRnaseq" : "RNASeq_mutationFileStrelkaSNV"
}

class VCFDataPreprocessor(object):
    
    def __init__(self, submissionfile):
        self.__clinicalData = pd.read_csv(submissionfile)
        self.__clinicalData["Patient Index"] = self.__clinicalData.index
        self.__clinicalData.index = self.__clinicalData["Patient"]
        self.__executor = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    
    def getPatientDataByDataset(self, directoryFolder='/test-data/'):
        result = {}
        reader = VCFReader()
        
        for dataset in GENOMIC_PROPS.keys():
            datasetDataframe = self.__clinicalData[self.__clinicalData[GENOMIC_PROPS[dataset]].notnull()]
            if not datasetDataframe.empty:
                data = PatientData(dataset, datasetDataframe.loc[datasetDataframe.index, "Patient"].copy())
                data = self.__fillClinicalData(data, datasetDataframe)
                
    
                filenames = datasetDataframe[GENOMIC_PROPS[dataset]].unique()
                paths = [ path.join(directoryFolder, f) for f in filenames]
                vcfgenescoredict = {}
                vcfgenesfunctiondict = {}
                for k, v in zip(filenames, self.__executor.map(reader.readVCFFileFindCompression, paths)):
                    vcfgenescoredict[k] = v[0]
                    vcfgenesfunctiondict[k] = v[1]
                
                vcfGenesScoreDF = self.__tranfromVCFDictToVCFDataframe(vcfgenescoredict, datasetDataframe, GENOMIC_PROPS[dataset])
                data.set_genes_scoring(vcfGenesScoreDF)
                vcfGenesFunctDF = self.__tranfromVCFDictToVCFDataframe(vcfgenesfunctiondict, datasetDataframe, GENOMIC_PROPS[dataset])
                data.set_genes_function_associated(vcfGenesFunctDF)
                result[dataset] = data
        
        return result
     
    def __fillClinicalData(self, patientdata, datasetDataframe):
        colums = self.__clinicalData.columns
        containsAge = "D_Age" in colums
        containsISS = "D_ISS" in colums
        containsHRFlag = "HR_FLAG" in colums
        
        if containsAge:
            ageDF = datasetDataframe.loc[datasetDataframe.index, "D_Age"].copy()
            for row in ageDF.index:
                ageDF.at[row] = self.__ageToGroup(ageDF[row])
            patientdata.set_ages(ageDF)
            
        if containsISS:
            patientdata.set_ISSs(datasetDataframe.loc[datasetDataframe.index, "D_ISS"].copy())
            
        if containsHRFlag:
            patientdata.set_flags(datasetDataframe.loc[datasetDataframe.index, "HR_FLAG"].copy())
            
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
            return 0;
        if age < 18:
            return 1;
        if age >=18 and age <35:
            return 2;
        if age >=35 and age <65:
            return 3;
        if age > 65:
            return 4;
        