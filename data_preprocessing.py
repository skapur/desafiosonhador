import os.path as path
import pandas as pd
import numpy as np
import multiprocessing
from readers.vcfreader import VCFReader

GCT_NAME = "sc2_Training_ClinAnnotations.csv"
SAMP_ID_NAME = "SamplId"

DATA_PROPS_EXPRESSION = {
    "MA": {
        "__dataparentfolder": "Expression Data",
        "__datafolder": "Microarray Data",
        "probe": "MA_probeLevelExpFile",
        "gene": "MA_geneLevelExpFile"
    },
    "RNASeq": {
        "__dataparentfolder": "Expression Data",
        "__datafolder": "RNA-Seq Data",
        "trans": "RNASeq_transLevelExpFile",
        "gene": "RNASeq_geneLevelExpFile"
        }
}

DATA_PROPS_GENOMIC = {
    "Genomic": {
        "__dataparentfolder": "Genomic Data",
        "__datafolder": "MMRF IA9 CelgeneProcessed",
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
    def __init__(self, parentFolder):
        self.__parentFolder = parentFolder
        self.clinicalData = pd.read_csv(path.join(self.__parentFolder, "Clinical Data", GCT_NAME))
        self.clinicalData["Patient Index"] = self.clinicalData.index
        self.clinicalData.index = self.clinicalData["Patient"]
        self.__executor = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        self.dataDict = None
        self.dataPresence = None

    def getData(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG"):
        type_level = DATA_PROPS_EXPRESSION[datype][level]
        type_level_sid = type_level + SAMP_ID_NAME
        baseCols = ["Patient", type_level, type_level_sid]
        subcd = self.clinicalData[baseCols + clinicalVariables + [outputVariable]].dropna(axis=0, subset=[type_level_sid])
        # print("Clinical data")
        # print(subcd.index.shape)
        dfiles = subcd[type_level].dropna().unique()
        dframes = [pd.read_csv(
            path.join(self.__parentFolder, DATA_PROPS_EXPRESSION[datype]["__dataparentfolder"], DATA_PROPS_EXPRESSION[datype]["__datafolder"],
                      dfile),
            index_col=[0], sep="," if dfile[-3:] == "csv" else "\t").T for dfile in dfiles]

        if len(dframes) > 1:
            df = pd.concat(dframes)
        else:
            df = dframes[0]

        # print("Dataframe:")
        # print(df.index.shape)
        df = df.loc[subcd[type_level_sid], :]
        df.index = subcd["Patient"]
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='any')
        return df, subcd.loc[df.index,clinicalVariables], subcd.loc[df.index,outputVariable]

    def getDataFrame(self, datype, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG",
                     savesubdataframe=""):
        type_level = DATA_PROPS_GENOMIC[datype][level]
        subdataset = self.clinicalData[["Patient", type_level["__csvIndex"]] + clinicalVariables + [outputVariable]]
        reader = VCFReader()
        pathdir = path.join(self.__parentFolder, DATA_PROPS_GENOMIC[datype]["__dataparentfolder"],
                            DATA_PROPS_GENOMIC[datype]["__datafolder"], type_level["__path"])
        filenames = self.clinicalData[type_level["__csvIndex"]].dropna().unique()
        paths = [path.join(pathdir, f) for f in filenames]
        vcfdict = {k: v for k, v in zip(filenames, self.__executor.map(reader.readVCFFile, paths))}
        vcfdataframe = pd.DataFrame(vcfdict)
        vcfdataframe = vcfdataframe.T
        vcfdataframe.fillna(value=0, inplace=True)
        if savesubdataframe:
            vcfdataframe.to_csv(savesubdataframe)
        subdataset.set_index(type_level["__csvIndex"], drop=False, append=False, inplace=True)
        subdataset = subdataset.join(vcfdataframe)

        subdataset = subdataset.loc[pd.notnull(subdataset.index)]
        subdataset.set_index("Patient", drop=True, append=False, inplace=True)
        # subdataset.index = subdataset["Patient"]
        subdataset = subdataset.drop(type_level["__csvIndex"], axis=1)
        return subdataset

    def getDataDict(self, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG"):
        return {(datype, level): self.getData(datype, level, clinicalVariables, outputVariable) for datype in
                DATA_PROPS_EXPRESSION.keys() for level in DATA_PROPS_EXPRESSION[datype] if "_" not in level}

    def generateDataDict(self):
        self.dataDict = self.getDataDict()
        self.dataPresence = self.__generateDataTypePresence()

    def assertDataDict(self):
        assert self.dataDict is not None, "Data dictionary must be generated before checking for data type presence"

    def __generateDataTypePresence(self):
        self.assertDataDict()
        return pd.DataFrame({pair: self.clinicalData.index.isin(df[0].index) for pair, df in self.dataDict.items()},
                            index=self.clinicalData.index)

    def modelPredictionMatrix(self, models={}):
        self.assertDataDict()


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
