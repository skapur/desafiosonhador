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
        'ALL_filtered' : {
            "__columnsDic" : "serialized_models/ALL_filtered_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/ALL_filtered_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/ALL_filtered_Classifier_CH1.pkl" 
            },
        'MuTectsnvs' : {
            "__columnsDic" : "serialized_models/MuTectsnvs_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/MuTectsnvs_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/MuTectsnvs_Classifier_CH1.pkl" 
            },
        'MuTectsnvs_filtered' : {
            "__columnsDic" : "serialized_models/MuTectsnvs_filtered_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/MuTectsnvs_filtered_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/MuTectsnvs_filtered_Classifier_CH1.pkl" 
            },
        'StrelkaIndels' : {
            "__columnsDic" : "serialized_models/StrelkaIndels_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/StrelkaIndels_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/StrelkaIndels_Classifier_CH1.pkl" 
            },
        'StrelkaIndels_filtered' : {
            "__columnsDic" : "serialized_models/StrelkaIndels_filtered_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/StrelkaIndels_filtered_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/StrelkaIndels_filtered_Classifier_CH1.pkl" 
            },
        'Strelkasnvs' : {
            "__columnsDic" : "serialized_models/Strelkasnvs_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/Strelkasnvs_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/Strelkasnvs_Classifier_CH1.pkl" 
            },
        'Strelkasnvs_filtered' : {
            "__columnsDic" : "serialized_models/Strelkasnvs_filtered_featColumns_CH1.pkl",
            "__transformerFilename" : "serialized_models/Strelkasnvs_filtered_Transformer_CH1.pkl",
            "__classifierFilename" : "serialized_models/Strelkasnvs_filtered_Classifier_CH1.pkl" 
            }        
        }
        self.__exploited_models = { 
            'MuTectRnaseq' : 'MuTectsnvs',
            'MuTectRnaseq_filtered' : 'MuTectsnvs_filtered',
            'StrelkaIndelsRnaseq' :'StrelkaIndels',
            'StrelkaIndelsRnaseq_filtered' :'StrelkaIndels_filtered',
            'StrelkasnvsRnaseq' :  'Strelkasnvs',
            'StrelkasnvsRnaseq_filtered' :  'Strelkasnvs_filtered'
        }
        self.__predictionModelToColumns = {
            "MuTectsnvs" : ["WES_mutationFileMutect"],
            "StrelkaIndels" : ["WES_mutationFileStrelkaIndel"],
            "Strelkasnvs" : ["WES_mutationFileStrelkaSNV"],
            "MuTectRnaseq" : ["RNASeq_mutationFileMutect"],
            "StrelkaIndelsRnaseq" : ["RNASeq_mutationFileStrelkaIndel"],
            "StrelkasnvsRnaseq" : ["RNASeq_mutationFileStrelkaSNV"],
            "ALL" : ["WES_mutationFileMutect", "WES_mutationFileStrelkaIndel", "WES_mutationFileStrelkaSNV", 
                "RNASeq_mutationFileMutect", "RNASeq_mutationFileStrelkaIndel", "RNASeq_mutationFileStrelkaSNV"],
            "MuTectsnvs_filtered" : ["WES_mutationFileMutect"],
            "StrelkaIndels_filtered" : ["WES_mutationFileStrelkaIndel"],
            "Strelkasnvs_filtered" : ["WES_mutationFileStrelkaSNV"],
            "MuTectRnaseq_filtered" : ["RNASeq_mutationFileMutect"],
            "StrelkaIndelsRnaseq_filtered" : ["RNASeq_mutationFileStrelkaIndel"],
            "StrelkasnvsRnaseq_filtered" : ["RNASeq_mutationFileStrelkaSNV"],
            "ALL_filtered" : ["WES_mutationFileMutect", "WES_mutationFileStrelkaIndel", "WES_mutationFileStrelkaSNV", 
                "RNASeq_mutationFileMutect", "RNASeq_mutationFileStrelkaIndel", "RNASeq_mutationFileStrelkaSNV"]
        }
    
    def generate_predictions_scores(self, dataset, modelType):
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
        dataset = dataset.loc[:, featColumns]
        print("Overlapping columns from prediction data for reducion")
        valuecounts = dataset.isnull().all().value_counts()
        print(valuecounts)
        print("Prediction Columns Size for reducion: " + str(len(dataset.columns)))
        
        x, z = self.__df_reduce(dataset, filename=self.__trained_Models[modelType]["__transformerFilename"])
        
        print("Reduced column size: " + str(len(z)))
        reducedDataset = dataset[dataset.columns[z]]
        print("Columns selected on reduced dataset: " + str(reducedDataset.columns))
        print("Finished reducing dataframe for prediction...")
        print("Dataset rows: " + str(len(dataset.index)))
        print("Starting to predict labels...")
        predictions = clf.predict(x)
        predictionscores = clf.predict_proba(x)
        scores =[]
        for i in range(0,len(predictions)):
            value = list(clf.classes_).index(predictions[i])
            scores.append(predictionscores[i, value])
        
        print("Finished to predict labels using model "+str(modelType)+"...")
        print("="*40)
        return predictions, scores
    
    def __df_reduce(self, X, filename = None):
        try: # load the objects from disk
            f = open(filename, 'rb')
            dic = pickle.load(f)
            inputer= dic['inputer']; variance = dic['variance']; scaler = dic['scaler']; fts = dic['fts']
            f.close()
            if inputer is not None:
                X = inputer.transform(X)
            if variance is not None:
                X = variance.transform(X)
            if scaler is not None:
                X = scaler.transform(X)
            if fts is not None:
                X = fts.transform(X)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise
        return X, fts.get_support(True)
    
    def generate_prediction_dataframe(self, clinicalDataframe, modelType, predictions, scores):
        print("Exporting prediction labels to file...")
        indexingdf = clinicalDataframe.dropna(
            subset=self.__predictionModelToColumns[modelType], 
            how='all')
        print("Exprected " + str(len(indexingdf)) + " rows predicted!")

        predicted = pd.DataFrame({"predictionscore":scores, "highriskflag":predictions}, index=indexingdf.index)
        predicted["highriskflag"] = predicted["highriskflag"].astype(str).apply(lambda x: x.upper())
        information = indexingdf[["Study", "Patient"]]
        outputDF = pd.concat([information, predicted], axis=1)
        outputDF = outputDF[["Study", "Patient", "predictionscore", "highriskflag"]]
        outputDF.columns = ["study", "patient", "predictionscore", "highriskflag"]
        outputDF = outputDF.reset_index(drop=True)
        return outputDF