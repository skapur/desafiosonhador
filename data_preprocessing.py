import os.path as path
import pandas as pd
import numpy as np
import multiprocessing
import sys
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
    }
}

GENOMIC_PROPS = {
    "MuTectsnvs" : "WES_mutationFileMutect",
    "StrelkaIndels" : "WES_mutationFileStrelkaIndel",
    "Strelkasnvs" : "WES_mutationFileStrelkaSNV"
}


def log_preprocessing(df):
    pass

def df_reduce(X, y, scaler = None, fts = None, fit = True, filename = None):
    import pickle
    if fit:
        scaler.fit(X, y); X = scaler.transform(X) 
        fts.fit(X, y); X = fts.transform(X)
        if filename is not None: # save the objects to disk
            f = open(filename, 'wb')
            pickle.dump({'scaler': scaler, 'fts': fts}, f)
            f.close()
    else:
        try: # load the objects from disk
            f = open(filename, 'rb')
            dic = pickle.load(f)
            scaler = dic['scaler']; fts = dic['fts']
            f.close()
            X = scaler.transform(X); X = fts.transform(X)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise
    return X, y, fts.get_support(True)


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
        subcd = self.clinicalData[baseCols + clinicalVariables + [outputVariable]].dropna(axis=0, subset=[type_level_sid])
        # print("Clinical data")
        # print(subcd.index.shape)
        dfiles = subcd[type_level].dropna().unique()
        dframes = [pd.read_csv(path.join(directoryFolder, dfile),
                               index_col=[0], sep = sep).T for dfile in dfiles]

        if len(dframes) > 1:
            df = pd.concat(dframes)
        else:
            df = dframes[0]

        df = df.loc[subcd[type_level_sid], :]
        df.index = subcd["Patient"]
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='any')
        return df, subcd.loc[df.index,clinicalVariables], subcd.loc[df.index,outputVariable]

    def getDataFrame(self, level, clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", savesubdataframe="", directoryFolder='/test-data/'):
        if outputVariable in clinicalVariables: clinicalVariables.remove(outputVariable)
        subdataset = self.clinicalData[["Patient", GENOMIC_PROPS[level]] + clinicalVariables + [outputVariable]]
        reader = VCFReader()
        filenames = self.clinicalData[GENOMIC_PROPS[level]].dropna().unique()
        if not filenames.size:
            return None
        paths = [ path.join(directoryFolder, f) for f in filenames]
        vcfdict =  { k : v for k, v in zip(filenames, self.__executor.map(reader.readVCFFile, paths))}
        vcfdataframe = pd.DataFrame(vcfdict)
        vcfdataframe = vcfdataframe.T
        vcfdataframe.fillna(value=0, inplace=True)

        subdataset.set_index(GENOMIC_PROPS[level], drop=False, append=False, inplace=True)
        subdataset = subdataset.join(vcfdataframe)

        subdataset = subdataset.loc[pd.notnull(subdataset.index)]
        subdataset.set_index("Patient", drop=True, append=False, inplace=True)
        #subdataset.index = subdataset["Patient"]
        subdataset = subdataset.drop(GENOMIC_PROPS[level], axis=1)
        if savesubdataframe:
            subdataset.to_csv(savesubdataframe)
        return subdataset
    
    def get_X_Y_FromDataframe(self, df, removeClinical=True, outputVariable="HR_FLAG"):
        df = df.fillna(value=0)
        if outputVariable == "HR_FLAG":
            df = df[df["HR_FLAG"] != "CENSORED"]
            y = df["HR_FLAG"] == "TRUE"
            x = df.drop("HR_FLAG", axis=1)
        else:
            x = df
            y = df[outputVariable]
    
        clinical = x[["D_Age", "D_ISS"]]
    
        if removeClinical:
            x = x.drop("D_Age", axis=1)
            x = x.drop("D_ISS", axis=1)
        return x, y, clinical
    
    def preprocessPrediction(self, useClinical=True, outputVariable="HR_FLAG", savePreprocessingDirectory='', directoryFolder='/test-data/'):
        muctectCSV = ''
        strelkaIndelsCSV = ''
        strelkasnvsCSV = ''
        if savePreprocessingDirectory:
            muctectCSV = path.join(savePreprocessingDirectory, "MuTectsnvs_joined.csv")
            strelkaIndelsCSV = path.join(savePreprocessingDirectory, "StrelkaIndels_joined.csv")
            strelkasnvsCSV = path.join(savePreprocessingDirectory, "Strelkasnvs_joined.csv")
            
        mucDF = self.getDataFrame("MuTectsnvs", outputVariable=outputVariable, savesubdataframe=muctectCSV, directoryFolder=directoryFolder)
        strelkaInDF = self.getDataFrame("StrelkaIndels", outputVariable=outputVariable, savesubdataframe=strelkaIndelsCSV, directoryFolder=directoryFolder)
        streklaSnDF = self.getDataFrame("Strelkasnvs", outputVariable=outputVariable, savesubdataframe=strelkasnvsCSV, directoryFolder=directoryFolder)
        
        if mucDF is not None and strelkaInDF is not None and streklaSnDF is not None:
            x, y, clinical = self.get_X_Y_FromDataframe(mucDF, outputVariable=outputVariable)

            x2, y2, clinical2 = self.get_X_Y_FromDataframe(strelkaInDF, outputVariable=outputVariable)
            x = pd.concat([x, x2], axis=1)
            y = pd.concat([y, y2])
            clinical = pd.concat([clinical, clinical2])
            
            x3, y3, clinical3 = self.get_X_Y_FromDataframe(streklaSnDF, outputVariable=outputVariable)
            x = pd.concat([x, x3], axis=1)
            y = pd.concat([y, y3])
            clinical = pd.concat([clinical, clinical3])
    
            x = x.groupby(x.columns, axis=1).sum()
            y = y.groupby(y.index).first()
            clinical = clinical.groupby(clinical.index).first()
            if useClinical:
                x = pd.concat([x, clinical], axis=1)
            
            return x, y, 'ALL'
        
        elif mucDF is not None:
            x, y, clinical = self.get_X_Y_FromDataframe(mucDF, outputVariable=outputVariable)
            if useClinical:
                x = pd.concat([x, clinical], axis=1)
            return x, y, 'MUC'
        
        elif strelkaInDF is not None and streklaSnDF is not None:
            x, y, clinical = self.get_X_Y_FromDataframe(strelkaInDF, outputVariable=outputVariable)
            x2, y2, clinical2 = self.get_X_Y_FromDataframe(streklaSnDF, outputVariable=outputVariable)
            x = pd.concat([x, x2], axis=1)
            y = pd.concat([y, y2])
            clinical = pd.concat([clinical, clinical2])
            
            x = x.groupby(x.columns, axis=1).sum()
            y = y.groupby(y.index).first()
            clinical = clinical.groupby(clinical.index).first()
            
            if useClinical:
                x = pd.concat([x, clinical], axis=1)
            
            return x, y, 'STR_ALL'
        
        elif strelkaInDF is not None:
            x, y, clinical = self.get_X_Y_FromDataframe(strelkaInDF, outputVariable)
            if useClinical:
                x = pd.concat([x, clinical], axis=1)
            return x, y, 'STR_IN'
        
        elif streklaSnDF is not None:
            x, y, clinical = self.get_X_Y_FromDataframe(strelkaInDF, outputVariable)
            if useClinical:
                x = pd.concat([x, clinical], axis=1)
            return x, y, 'STR_SN'
        else:
            print('The input challenge file is not been read correctly!')
            return None, None, None

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