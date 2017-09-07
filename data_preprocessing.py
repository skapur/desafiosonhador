import os.path as path
import pandas as pd
import numpy as np

GCT_NAME = "sc2_Training_ClinAnnotations.csv"
#GCT_NAME = "globalClinTraining.csv"
SAMP_ID_NAME = "SamplId"
DATA_PROPS = {
    "MA": {
        "__folder": "Microarray Data",
        "probe": "MA_probeLevelExpFile",
        "gene": "MA_geneLevelExpFile"
    },
    "RNASeq": {
        "__folder": "RNA-Seq Data",
        "trans": "RNASeq_transLevelExpFile",
        "gene": "RNASeq_geneLevelExpFile"
    }
}


def log_preprocessing(df):
    pass


class MMChallengeData(object):
    def __init__(self, parentFolder):
        self.__parentFolder = parentFolder
        self.clinicalData = pd.read_csv(path.join(self.__parentFolder, "Clinical Data", GCT_NAME))
        self.clinicalData["Patient Index"] = self.clinicalData.index
        self.clinicalData.index = self.clinicalData["Patient"]
        self.dataDict = None
        self.dataPresence = None

    def getData(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG"):

        type_level = DATA_PROPS[datype][level]
        type_level_sid = type_level + SAMP_ID_NAME
        baseCols = ["Patient", type_level, type_level_sid]
        subcd = self.clinicalData[baseCols + clinicalVariables + [outputVariable]].dropna(subset=baseCols)
        dfiles = subcd[type_level].dropna().unique()
        print(dfiles)
        dframes = [pd.read_csv(path.join(self.__parentFolder, "Expression Data", DATA_PROPS[datype]["__folder"], dfile),
                               index_col=[0], sep = "," if "csv" in dfile[-3:] else "\t").T for dfile in dfiles]

        if len(dframes) > 1:
            df = pd.concat(dframes)
        else:
            df = dframes[0]

        df = df.loc[subcd[type_level_sid], :]
        df.index = subcd["Patient"]
        return df, subcd[clinicalVariables], subcd[outputVariable]

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

    def modelPredictionMatrix(self, models={}):
        self.assertDataDict()

class MMChallengePredictor(object):

    def __init__(self, mmc_data, mmc_data_presence, clf, data_types, single_vector_apply_fun=lambda x: x, multiple_vector_apply_fun=lambda x: x,name="Default Model"):
        self.data_dict = mmc_data
        self.data_presence = mmc_data_presence
        self.clf = clf
        assert not False in [dty in self.data_dict.keys() for dty in data_types], "Data types must exist on the data dictionary"
        self.data_types = data_types
        self.vapply = single_vector_apply_fun
        self.capply = multiple_vector_apply_fun

    def predict_case(self, index):
        hasCorrectData = self.data_presence.loc[index,self.data_types]
        #print(hasCorrectData)
        if hasCorrectData.all():
            X = self.get_feature_vector(index)
            return self.clf.predict(X)
        else:
            return np.nan, np.nan

    def get_feature_vector(self, index):
        frame = None
        if len(self.data_types) > 1:
            frame = self.capply(pd.concat([self.vapply(self.data_dict[dty][0].loc[index,:]) for dty in self.data_types], axis=0))
        elif len(self.data_types) == 1:
            frame = self.vapply(self.data_dict[self.data_types[0]][0].loc[index,:])
        else:
            raise Exception("Data type tuple must contain at least one element")
        return frame
