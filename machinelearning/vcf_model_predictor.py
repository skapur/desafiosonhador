import sys
import pickle
import pandas as pd

class VCFModelPredictor(object):
    
    def __init__(self):
        self.__trained_Models = {
        'ALL' : {
            "__columnsDic" : "serialized_models/ALL_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/ALL_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/ALL_Classifier_CH1.pkl" 
            },
        'MuTectsnvs' : {
            "__columnsDic" : "serialized_models/MuTectsnvs_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/MuTectsnvs_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/MuTectsnvs_Classifier_CH1.pkl" 
            },
        'StrelkaIndels' : {
            "__columnsDic" : "serialized_models/StrelkaIndels_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/StrelkaIndels_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/StrelkaIndels_Classifier_CH1.pkl" 
            },
        'Strelkasnvs' : {
            "__columnsDic" : "serialized_models/Strelkasnvs_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/Strelkasnvs_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/Strelkasnvs_Classifier_CH1.pkl" 
            }          
        }
        self.__exploited_models = { 
            'MuTectRnaseq' : 'MuTectsnvs',
            'StrelkaIndelsRnaseq' :'StrelkaIndels',
            'StrelkasnvsRnaseq' :  'Strelkasnvs'
        }
    
    
    def generate_predictions_scores(self, x, modelType):
        if modelType in self.__exploited_models.keys():
            modelType = self.__exploited_models[modelType]
        
        print("Starting reading model Files...")
        f = open(self.__trained_Models[modelType]["__columnsDic"], 'rb')
        featColumns = pickle.load(f);
        f.close();
        
        f = open(self.__trained_Models[modelType]["__classifierFilename"], 'rb')
        clf = pickle.load(f)
        f.close();
        
        print("Finished reading model Files...")
        print("Starting reducing dataframe for prediction...")
        x = x.loc[:, featColumns]
        print("Overlapping columns from prediction data")
        print(x.isnull().all())
        x = x.fillna(value=0)
        
        x, z = self.__df_reduce(x, filename=self.__trained_Models[modelType]["__transformerFilename"])
        
        print("Finished reducing dataframe for prediction...")
        print("Starting to predict labels...")
        predictions = clf.predict(x)
        scores = clf.predict_proba(x)[:,1]
        print("Finished to predict labels using model "+str(modelType)+"...")
        
        return predictions, scores
    
    def __df_reduce(self, X, filename = None):
        try: # load the objects from disk
            f = open(filename, 'rb')
            dic = pickle.load(f)
            scaler = dic['scaler']; fts = dic['fts']
            f.close()
            if scaler is not None:
                X = scaler.transform(X)
            if fts is not None:
                X = fts.transform(X)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise
        return X, fts.get_support(True)
    
    def generate_prediction_dataframe(self, clinicalDataframe, predictions, scores):
        print("Exporting prediction labels to file...")
        indexingdf = clinicalDataframe.dropna(
            subset=["WES_mutationFileMutect", "WES_mutationFileStrelkaIndel", "WES_mutationFileStrelkaSNV", 
                "RNASeq_mutationFileMutect", "RNASeq_mutationFileStrelkaIndel", "RNASeq_mutationFileStrelkaSNV"], 
            how='all')
        
        print("There are " + str(len(clinicalDataframe.index) - len(indexingdf.index)) + " patient rows wihout any WES file, they will be discarded from predictions...")
        predicted = pd.DataFrame({"predictionscore":scores, "highriskflag":predictions}, index=indexingdf.index)
        information = indexingdf[["Study", "Patient"]]
        outputDF = pd.concat([information, predicted], axis=1)
        outputDF = outputDF[["Study", "Patient", "predictionscore", "highriskflag"]]
        outputDF.columns = ["study", "patient", "predictionscore", "highriskflag"]
        return outputDF