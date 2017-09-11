import os.path as path
import pandas as pd
import numpy as np
import multiprocessing
from readers.vcfreader import VCFReader

SAMP_ID_NAME = "SamplId"
DATA_PROPS = {
    "MA": {
        "probe": "MA_probeLevelExpFile",
        "gene": "MA_geneLevelExpFile"
    },
    "RNASeq": {
        "trans": "RNASeq_transLevelExpFile",
        "gene": "RNASeq_geneLevelExpFile"
    },
    "Genomic": {
        "MuTectsnvs": {
                "__path": "MuTect2 SnpSift Annotated vcfs",
                "__csvIndex": "WES_mutationFileMutect"
                },
        "StrelkaIndels": {
                "__path": "Strelka SnpSift Annotated vcfs/indels",
                "__csvIndex": "WES_mutationFileStrelkaIndel"
            },
        "Strelkasnvs": {
            "__path": "Strelka SnpSift Annotated vcfs/snvs",
            "__csvIndex": "WES_mutationFileStrelkaSNV"
            }
    }
}


def log_preprocessing(df):
    pass


class MMChallengeData(object):
    def __init__(self, submissionfile):
        self.clinicalData = pd.read_csv(submissionfile)
        self.clinicalData["Patient Index"] = self.clinicalData.index
        self.clinicalData.index = self.clinicalData["Patient"]
        self.__executor = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        self.dataDict = None
        self.dataPresence = None

    def getData(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", sep=",", directoryFolder='/test-data/'):
        type_level = DATA_PROPS[datype][level]
        type_level_sid = type_level + SAMP_ID_NAME
        baseCols = ["Patient", type_level, type_level_sid]
        subcd = self.clinicalData[baseCols + clinicalVariables + [outputVariable]].dropna(subset=baseCols)
        dfiles = subcd[type_level].dropna().unique()
        print(dfiles)
        dframes = [pd.read_csv(path.join(directoryFolder, dfile),
                               index_col=[0], sep = sep).T for dfile in dfiles]

        if len(dframes) > 1:
            df = pd.concat(dframes)
        else:
            df = dframes[0]

        df = df.loc[subcd[type_level_sid], :]
        df.index = subcd["Patient"]
        return df, subcd[clinicalVariables], subcd[outputVariable]

    def getDataFrame(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", savesubdataframe="", directoryFolder='/test-data/'):
        type_level = DATA_PROPS[datype][level]
        subdataset = self.clinicalData[["Patient", type_level["__csvIndex"]] + clinicalVariables + [outputVariable]]
        reader = VCFReader()
        filenames = self.clinicalData[type_level["__csvIndex"]].dropna().unique()
        paths = [ path.join(directoryFolder, f) for f in filenames]
        vcfdict =  { k : v for k, v in zip(filenames, self.__executor.map(reader.readVCFFile, paths))}
        vcfdataframe = pd.DataFrame(vcfdict)
        vcfdataframe = vcfdataframe.T
        vcfdataframe.fillna(value=0, inplace=True)
        if savesubdataframe:
            vcfdataframe.to_csv(savesubdataframe)
        subdataset.set_index(type_level["__csvIndex"], drop=False, append=False, inplace=True)
        subdataset = subdataset.join(vcfdataframe)

        subdataset = subdataset.loc[pd.notnull(subdataset.index)]
        subdataset.set_index("Patient", drop=True, append=False, inplace=True)
        #subdataset.index = subdataset["Patient"]
        subdataset = subdataset.drop(type_level["__csvIndex"], axis=1)
        return subdataset

    def getDataDict(self, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG"):
        return {(datype, level): self.getData(datype, level, clinicalVariables, outputVariable) for datype in DATA_PROPS.keys() for level in DATA_PROPS[datype] if "_" not in level}
        
    def generateDataDict(self):
        self.dataDict = self.getDataDict()
        self.dataPresence = self.__generateDataTypePresence()
        
    def assertDataDict(self):
        assert self.dataDict is not None, "Data dictionary must be generated before checking for data type presence"
        
    def __generateDataTypePresence(self):
        self.assertDataDict()
        return pd.DataFrame({pair: self.clinicalData.index.isin(df[0].index) for pair, df in self.dataDict.items()},index=self.clinicalData.index)

class MMChallengePredictor(object):

    def predict_case(self, index):
        hasCorrectData = self.data_presence.loc[index,self.data_types]
        #print(hasCorrectData)
        if hasCorrectData.all():
            X = self.get_feature_vector(index)
            return self.predict_fun(X), self.confidence_fun(X)
        else:
            return np.nan, np.nan
        
    def get_pred_df_row(self,case):
        row = self.clinical_data.loc[case,:][["Study","Patient"]].tolist()
        try:
            flag, score = self.predict_case(case)
        except Exception as e:
            flag, score = np.nan, np.nan
        row = row + [flag,score]
        return row
    
    def predict_dataset(self):
        columns = ["study","patient",'_'.join(["predictionscore",self.predictor_name]),'_'.join(["highriskflag",self.predictor_name])]
        rows = [self.get_pred_df_row(case) for case in self.clinical_data.index]
        df = pd.DataFrame(rows)
        df.columns = columns
        return pd.DataFrame(df)
    
    def get_feature_vector(self, index):
        frame = None
        if len(self.data_types) > 1:
            frame = self.capply(pd.concat([self.vapply(self.data_dict[dty][0].loc[index,:]) for dty in self.data_types], axis=0))
        elif len(self.data_types) == 1:
            frame = self.vapply(self.data_dict[self.data_types[0]][0].loc[index,:])
        else:
            raise Exception("Data type tuple must contain at least one element")
        return frame