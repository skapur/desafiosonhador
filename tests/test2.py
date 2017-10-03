import pickle

from readers.vcfreader import VCFReader

import os.path as path
import pandas as pd
import numpy as np


GENOMIC_PROPS = {
    "MuTectsnvs" : "WES_mutationFileMutect",
    "StrelkaIndels" : "WES_mutationFileStrelkaIndel",
    "Strelkasnvs" : "WES_mutationFileStrelkaSNV",
    "MuTectRnaseq" : "RNASeq_mutationFileMutect",
    "StrelkaIndelsRnaseq" : "RNASeq_mutationFileStrelkaIndel",
    "StrelkasnvsRnaseq" : "RNASeq_mutationFileStrelkaSNV"
}

def loadcolumns():
    f = open("/home/tiagoalves/rrodrigues/desafiosonhador/serialized_models/ALL_featColumns_CH1.pkl", 'rb')
    featColumns = pickle.load(f);
    print(featColumns)
    f.close();
    df = pd.read_csv("/home/tiagoalves/Downloads/MMfiles/MutatedGenes.txt", sep="\t")
    print(df.shape)
    genes = df["Gene"]
    intersected = set(genes) & set(featColumns)
    print(len(intersected))

def getAllFunctions():
    submissionfile = '/home/tiagoalves/rrodrigues/globalClinTraining.csv'
    directoryFolder ='/home/tiagoalves/rrodrigues/link-data/'
    clinicalData = pd.read_csv(submissionfile)
    clinicalData["Patient Index"] = clinicalData.index
    clinicalData.index = clinicalData["Patient"]
    filenames = None
    levels =["WES_mutationFileMutect", "WES_mutationFileStrelkaIndel", "WES_mutationFileStrelkaSNV", 
             "RNASeq_mutationFileMutect", "RNASeq_mutationFileStrelkaIndel", "RNASeq_mutationFileStrelkaSNV"]
    for level in levels:
        if filenames is None:
            filenames = clinicalData[[level, "HR_FLAG"]].dropna()
            filenames.index = filenames[level]
        else:
            df = clinicalData[[level, "HR_FLAG"]].dropna()
            df.index = df[level]
            filenames = pd.concat([filenames,df], axis=0)
    print(len(filenames))
    paths = [ path.join(directoryFolder, f) for f in filenames.index]
    reader = VCFReader()
    print("reading")
    f = open("/home/tiagoalves/rrodrigues/outputinfo.txt", 'w')
    reader.getAllFunctionsWithTrueAndFalses(paths, filenames["HR_FLAG"], f)
    print("Finished")
    f.close()
    
def getGenesAndFunctions():
    outputFile = '/home/tiagoalves/rrodrigues/outputinfo.txt'
    with open(outputFile) as f:
        content = f.readlines()

    onlyTrue = content[1].strip()
    onlyTrue = set(onlyTrue[1:-1].split(","))
    onlyfalse = content[3].strip()[1:-1].split(",")
    for z in onlyfalse:
        onlyTrue.add(z)
    genes = set()
    functions = set()
    for entry in onlyTrue:
        splited = entry.strip()[1:-1].split("_",1)
        genes.add(splited[0])
        functions.add(splited[1])
    
    f = open("/home/tiagoalves/rrodrigues/filteringGenesAndFunctions.pkl", "wb")
    pickle.dump({"genes": genes, "functions": functions}, f)
    f.close()
    
    #print(genes)
    #print(functions)
    

if __name__ == '__main__':
    getGenesAndFunctions()