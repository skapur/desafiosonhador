import sys
sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')
from data_preprocessing import MMChallengeData
import os.path as path
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, log_loss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def create_csv_from_data():
    patha = '/home/dreamchallenge/synapse/syn7222203';
    mmcd = MMChallengeData(patha)
    df = mmcd.getDataFrame("Genomic", "MuTectsnvs", savesubdataframe='/home/dreamchallenge/synapse/syn7222203/MuTectsnvs.csv')
    df.to_csv("/home/dreamchallenge/synapse/syn7222203/MuTectsnvs_joined.csv")

def create_csv_from_data2():
    df1 = DataFrame.from_csv('/home/tiagoalves/rrodrigues/globalClinTraining.csv')
    df1 = df1[["WES_mutationFileMutect", "D_Age", "D_ISS","HR_FLAG"]]
    df2 = DataFrame.from_csv('/home/tiagoalves/rrodrigues/MuTectsnvs.csv')
    df1.set_index("WES_mutationFileMutect", drop=True, append=False, inplace=True)
    df3 = df1.join(df2)
    df4 = df3.loc[pd.notnull(df3.index)]
    df5 = df4[df4["HR_FLAG"] != "CENSORED"]
    df5.to_csv("/home/tiagoalves/rrodrigues/MuTectsnvs_joined.csv")

def prepair_dataframe(dataframe):
    dataframe = dataframe.fillna(value=0)
    if "WES_mutationFileMutect" in dataframe.columns:
        dataframe = dataframe.drop("WES_mutationFileMutect", axis=1)
    if "WES_mutationFileStrelkaIndel" in dataframe.columns:
        dataframe = dataframe.drop("WES_mutationFileStrelkaIndel", axis=1)
    if "WES_mutationFileStrelkaSNV" in dataframe.columns:
        dataframe = dataframe.drop("WES_mutationFileStrelkaSNV", axis=1)
    return dataframe

def report(cvr):
    for name, array in cvr.items():
        print(name+" : "+str(np.mean(array))+" +/- "+str(np.std(array)))

def run_traning():
    df = DataFrame.from_csv("/home/tiagoalves/rrodrigues/MuTectsnvs_joined.csv")
    df = prepair_dataframe(df)
    y = df["HR_FLAG"]
    x = df.drop("HR_FLAG", axis=1)
    anova_filter = SelectKBest(f_regression, k=5)
    clf = svm.SVC(kernel='linear', probability=True)
    anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
    anova_svm.set_params(anova__k=10, svc__C=.1)
    #pipeline_factory = lambda clf: Pipeline(steps=[("scale", StandardScaler()), ("feature_selection",SelectPercentile(percentile=30)), ("classify", clf)])
    cv = cross_validate(anova_svm, x, y, scoring=["accuracy","recall","f1","neg_log_loss","precision"], cv = 10)
    report(cv)
    
if __name__ == '__main__':
    create_csv_from_data()
    
    

    