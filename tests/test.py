import sys
sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')
from data_preprocessing import MMChallengeData, df_reduce
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.neural_network import MLPClassifier
import pickle
from initial_file_telma import preprocess, featureSelection, modelTrain

def create_csv_from_data():
    patha = '/home/dreamchallenge/synapse/syn7222203/Clinical Data/globalClinTraining.csv';
    mmcd = MMChallengeData(patha)
    x,y, modeltype = mmcd.preprocessPrediction(savePreprocessingDirectory = "/home/dreamchallenge/", directoryFolder="/home/dreamchallenge/link-data/")
    print(modeltype)
    df = pd.concat([x, y], axis=1)
    df.to_csv("/home/dreamchallenge/synapse/ALL_joined.csv")

def create_csv_from_data2():
    df1 = DataFrame.from_csv('/home/tiagoalves/rrodrigues/globalClinTraining.csv')
    df1 = df1[["WES_mutationFileStrelkaIndel", "Patient", "D_Age", "D_ISS","HR_FLAG"]]
    df2 = DataFrame.from_csv('/home/tiagoalves/rrodrigues/StrelkaIndels.csv')
    df1.set_index("WES_mutationFileStrelkaIndel", drop=True, append=False, inplace=True)
    df3 = df1.join(df2)
    df4 = df3.loc[pd.notnull(df3.index)]
    df5 = df4[df4["HR_FLAG"] != "CENSORED"]
    df5.set_index("Patient", drop=True, append=False, inplace=True)
    df5.to_csv("/home/tiagoalves/rrodrigues/StrelkaIndels_joined.csv")

def report(cvr):
    for name, array in cvr.items():
        print(name+" : "+str(np.mean(array))+" +/- "+str(np.std(array)))

def prepare_dataframe(dataframe, removeClinical=False):
    df = DataFrame.from_csv(dataframe)
    df = df.fillna(value=0)
    
    df = df[df["HR_FLAG"] != "CENSORED"]
    y = df["HR_FLAG"] == "TRUE"
    x = df.drop("HR_FLAG", axis=1)
    
    clinical = x[["D_Age", "D_ISS"]]
    
    if removeClinical:
        x = x.drop("D_Age", axis=1)
        x = x.drop("D_ISS", axis=1)
    return x, y, clinical

def run_traning(dataframefilename, removeClinical=False):
    x, y = prepare_dataframe(dataframefilename, removeClinical)
    
    x = preprocess(x, 'scaler')
    x = featureSelection(x, y, percentile = 40)
    
    models = ['knn', 'nbayes', 'decisionTree', 'logisticRegression', 'svm',
              'nnet', 'rand_forest', 'bagging']

    # Test models
    for model in models:
        modelTrain(x, y, method = model)

def joindataframes(dataframefiles, useClinical=False, saveToFile=''):
    x = None
    y = None
    clinical = None
    for file in dataframefiles:
        x1, y1, clinical1 = prepare_dataframe(file, True)
        if x is None and y is None and clinical is None:
            x, y, clinical = x1, y1, clinical1
        else:
            x = pd.concat([x, x1], axis=1)
            y = pd.concat([y, y1])
            clinical = pd.concat([clinical, clinical1])
    
    x = x.groupby(x.columns, axis=1).sum()
    y = y.groupby(y.index).first()
    clinical = clinical.groupby(clinical.index).first()
    
    if useClinical:
        x = pd.concat([x, clinical], axis=1)
    
    if saveToFile:
        z = pd.concat([x,clinical], axis=1)
        z = pd.concat([z,y], axis=1)
        z.to_csv(saveToFile)
    
    return x,y
        
def run_traning_joiningFiles(dataframefiles, useClinical=False, saveToFile='', doCV=True, saveTransformerFile='', saveClassifierFile=''):
    #processingData = MMChallengeData(dataframefiles);
    #x, y, modeltype = processingData.preprocessPrediction(directoryFolder ="/home/tiagoalves/rrodrigues/");
    #print(modeltype)
    x, y = joindataframes(dataframefiles, useClinical, saveToFile)
    x = preprocess(x, 'scaler')
    x = featureSelection(x, y, percentile = 40)

    scaler = StandardScaler()
    fts = SelectPercentile(percentile=40)
    x, y, z = df_reduce(x, y, scaler, fts, True, saveTransformerFile)

    if doCV:
        #models = ['knn', 'nbayes', 'decisionTree', 'logisticRegression', 'svm', 'nnet', 'rand_forest', 'bagging']
        models = ['nnet']

        # Test models
        for model in models:
            modelTrain(x, y, method = model)
    
    if saveTransformerFile:
        clf = MLPClassifier(solver = 'lbfgs', activation = "logistic", hidden_layer_sizes = (250,), alpha = 0.001)
        clf.fit(x, y)
        f = open(saveClassifierFile, 'wb')
        pickle.dump(clf,f)
        f.close()
        
    
if __name__ == '__main__':
    create_csv_from_data()
    #run_traning("/home/tiagoalves/rrodrigues/Strelkasnvs_joined.csv", removeClinical=True)
    '''
    dataframefiles = ['/home/tiagoalves/rrodrigues/MuTectsnvs_joined.csv',
                      '/home/tiagoalves/rrodrigues/Strelkasnvs_joined.csv', 
                      '/home/tiagoalves/rrodrigues/StrelkaIndels_joined.csv']
              
    '''
    #dataframefiles = ['/home/tiagoalves/rrodrigues/StrelkaIndels_joined.csv']
    #dataframefiles = '/home/tiagoalves/rrodrigues/test.csv'
    '''
    run_traning_joiningFiles(dataframefiles, useClinical=True, doCV=True, 
                             saveTransformerFile='/home/tiagoalves/rrodrigues/ALL_Transformer_CH1.pkl',
                             saveClassifierFile='/home/tiagoalves/rrodrigues/ALL_Classifier_CH1.pkl')
    '''
    

    