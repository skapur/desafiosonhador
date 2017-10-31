import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
    submissionfile = '/home/dreamchallenge/link-data/globalClinTraining.csv'
    directoryFolder ='/home/dreamchallenge/link-data/'
    clinicalData = pd.read_csv(submissionfile)
    clinicalData.replace({'.FILTERED': ''}, regex=True, inplace=True)
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
    f = open("/home/dreamchallenge/outputinfo.txt", 'w')
    reader.getAllFunctionsWithTrueAndFalses(paths, filenames["HR_FLAG"], f)
    print("Finished")
    f.close()
    
def getGenesAndFunctions():
    outputFile = '/home/dreamchallenge/outputinfo.txt'
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
    
    f = open("/home/dreamchallenge/filteringGenesAndFunctions.pkl", "wb")
    pickle.dump({"genes": genes, "functions": functions}, f)
    f.close()
    
    #print(genes)
    #print(functions)

def tLODTest():
    reader = VCFReader()
    genes = reader.getGenesWithUpperTLOD("/home/tiagoalves/rrodrigues/resources/synapse/MMRF_1021_1_BM_CD138pos_T2_KAS5U_L02366.MarkDuplicates.mdup...ANNOTATED.vcf.gz")
    print(genes)
    
def qsiTest():
    reader = VCFReader()
    genes = reader.getGenesWithUpperQSI("/home/tiagoalves/rrodrigues/resources/synapse/MMRF_1037_1_BM_CD138pos_T2_TSE61_K02458.MarkDuplicates.mdup.seqvar.all.somatic.snvs.ANNOTATED.vcf.gz")
    print(genes)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.join(os.path.dirname(__file__), '../..'))
    os.chdir(dir_path)
    #getAllFunctions()
    #getGenesAndFunctions()
    #tLODTest()
    qsiTest()