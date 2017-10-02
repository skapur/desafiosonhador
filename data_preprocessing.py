import os.path as path
import pandas as pd
import numpy as np

SAMP_ID_NAME = "SamplId"
DATA_PROPS = {
    "MA": {
        "probe": "MA_probeLevelExpFile",
        "gene": "MA_geneLevelExpFile"
    },
    "RNASeq": {
        "trans": "RNASeq_transLevelExpFile",
        "gene": "RNASeq_geneLevelExpFile"
    }
}

def log_preprocessing(df):
    pass

class MMChallengeData(object):
    def __init__(self, submissionfile):
        self.clinicalData = pd.read_csv(submissionfile)
        self.clinicalData["Patient Index"] = self.clinicalData.index
        self.clinicalData.index = self.clinicalData["Patient"]
        self.dataDict = None
        self.dataPresence = None

    def getData(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG",
                directoryFolder='/test-data/', columnNames=None, NARemove=[True, True], colParseFun=None):
        type_level = DATA_PROPS[datype][level]
        type_level_sid = type_level + SAMP_ID_NAME
        baseCols = ["Patient", type_level, type_level_sid]
        subcd = self.clinicalData[baseCols + clinicalVariables + [outputVariable]].dropna(axis=0,
                                                                                          subset=[type_level_sid])

        dfiles = subcd[type_level].dropna().unique()
        dframes = [pd.read_csv(path.join(directoryFolder, dfile),
                               index_col=[0], sep="," if dfile[-3:] == "csv" else "\t").T for dfile in dfiles]

        if len(dframes) > 1:
            df = pd.concat(dframes)
        else:
            df = dframes[0]
        #print(df.columns.tolist())
        df.columns = [colParseFun(col) if colParseFun is not None else col for col in df.columns.tolist()]
        df = df.loc[subcd[type_level_sid], :]
        df.index = subcd["Patient"]
        removeCols = NARemove[1]
        removeRows = NARemove[0]
        if columnNames is None:
            df = df.dropna(axis=0, how='all') if removeRows else df
            df = df.dropna(axis=1, how='any') if removeCols else df
        else:
            df = df.loc[:, columnNames].fillna(value=0)

        return df, subcd.loc[df.index, clinicalVariables], subcd.loc[df.index, outputVariable]

    def getDataDict(self, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", directoryFolder='/test-data/',
                    columnNames=None, NARemove=[True, True], colParseFunDict=None):
        return {(datype, level): self.getData(
            datype,
            level,
            clinicalVariables,
            outputVariable,
            directoryFolder,
            columnNames[(datype, level)] if columnNames is not None else None,
            NARemove,
            None if colParseFunDict is None else colParseFunDict[(datype, level)] if (datype, level) in colParseFunDict.keys() else lambda x:x)
            for datype in DATA_PROPS.keys() for level in DATA_PROPS[datype] if "_" not in level}

    def generateDataDict(self, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG",
                         directoryFolder='/test-data/', columnNames=None, NARemove=[True, True], colParseFunDict=None):
        self.dataDict = self.getDataDict(clinicalVariables, outputVariable, directoryFolder, columnNames, NARemove, colParseFunDict)
        self.dataPresence = self.__generateDataTypePresence()

    def assertDataDict(self):
        assert self.dataDict is not None, "Data dictionary must be generated before checking for data type presence"

    def __generateDataTypePresence(self):
        self.assertDataDict()
        return pd.DataFrame({pair: self.clinicalData.index.isin(df[0].index) for pair, df in self.dataDict.items()},
                            index=self.clinicalData.index)


class MMChallengePredictor(object):
    def __init__(self, mmcdata, predict_fun, confidence_fun, data_types, single_vector_apply_fun=lambda x: x,
                 multiple_vector_apply_fun=lambda x: x, predictor_name="Default Predictor"):
        self.data_dict = mmcdata.dataDict
        self.data_presence = mmcdata.dataPresence
        self.clinical_data = mmcdata.clinicalData
        self.predictor_name = predictor_name
        self.predict_fun = predict_fun
        self.confidence_fun = confidence_fun

        assert not False in [dty in self.data_dict.keys() for dty in
                             data_types], "Data types must exist on the data dictionary"
        self.data_types = data_types
        self.vapply = single_vector_apply_fun
        self.capply = multiple_vector_apply_fun

    def predict_case(self, index):
        hasCorrectData = self.data_presence.loc[index, self.data_types]
        # print(hasCorrectData)
        if hasCorrectData.all():
            X = self.get_feature_vector(index)
            return self.predict_fun(X), self.confidence_fun(X)
        else:
            return np.nan, np.nan

    def get_pred_df_row(self, case):
        row = self.clinical_data.loc[case, :][["Study", "Patient"]].tolist()
        try:
            flag, score = self.predict_case(case)
        except Exception as e:
            flag, score = np.nan, np.nan
        row = row + [score, flag]
        return row

    def predict_dataset(self):
        columns = ["study", "patient", '_'.join(["predictionscore", self.predictor_name]),
                   '_'.join(["highriskflag", self.predictor_name])]
        rows = [self.get_pred_df_row(case) for case in self.clinical_data.index]
        df = pd.DataFrame(rows)
        df.columns = columns
        return pd.DataFrame(df)

    def get_feature_vector(self, index):
        frame = None
        if len(self.data_types) > 1:
            frame = self.capply(
                pd.concat([self.vapply(self.data_dict[dty][0].loc[index, :]) for dty in self.data_types], axis=0))
        elif len(self.data_types) == 1:
            frame = self.vapply(self.data_dict[self.data_types[0]][0].loc[index, :])
        else:
            raise Exception("Data type tuple must contain at least one element")
        return frame
