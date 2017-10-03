#!/usr/bin/python

import sys, getopt

from machinelearning.vcf_model_predictor import VCFModelPredictor
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
import pandas as pd

joinALLDatasets = False

trained_Models = {
    'ALL' : {
        "__columnsDic" : "/desafiosonhador/serialized_models/ALL_featColumns_CH1.pkl",
        "__transformerFilename" : "/desafiosonhador/serialized_models/ALL_Transformer_CH1.pkl",
        "__classifierFilename" : "/desafiosonhador/serialized_models/ALL_Classifier_CH1.pkl" 
        },
    'MUC' : {
        "__columnsDic" : "/desafiosonhador/serialized_models/MuTectsnvs_featColumns_CH1.pkl",
        "__transformerFilename" : "/desafiosonhador/serialized_models/MuTectsnvs_Transformer_CH1.pkl",
        "__classifierFilename" : "/desafiosonhador/serialized_models/MuTectsnvs_Classifier_CH1.pkl" 
        },
    'STR_ALL' : {
        "__columnsDic" : "/desafiosonhador/serialized_models/Strelka_featColumns_CH1.pkl",
        "__transformerFilename" : "/desafiosonhador/serialized_models/Strelka_Transformer_CH1.pkl",
        "__classifierFilename" : "/desafiosonhador/serialized_models/Strelka_Classifier_CH1.pkl" 
        },
    'STR_IN' : {
        "__columnsDic" : "/desafiosonhador/serialized_models/StrelkaIndels_featColumns_CH1.pkl",
        "__transformerFilename" : "/desafiosonhador/serialized_models/StrelkaIndels_Transformer_CH1.pkl",
        "__classifierFilename" : "/desafiosonhador/serialized_models/StrelkaIndels_Classifier_CH1.pkl" 
        },
    'STR_SN' : {
        "__columnsDic" : "/desafiosonhador/serialized_models/Strelkasnvs_featColumns_CH1.pkl",
        "__transformerFilename" : "/desafiosonhador/serialized_models/Strelkasnvs_Transformer_CH1.pkl",
        "__classifierFilename" : "/desafiosonhador/serialized_models/Strelkasnvs_Classifier_CH1.pkl" 
        }          
    }

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('subchallenge1.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('subchallenge1.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    
    print("Starting reading VCF Files...")
    preprocessor = VCFDataPreprocessor(inputfile)
    datasets = preprocessor.getPatientDataByDataset()
    print("Finished reading VCF Files...")
    
    predictor = VCFModelPredictor()
    
    if not joinALLDatasets:
        predictedDFs = []
        for modelType in datasets.keys():
            X = datasets[modelType].getFullDataframe(False, False)
            predictions, scores = predictor.generate_predictions_scores(X, modelType)
            predictedDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), predictions, scores)
            predictedDF.set_index("patient", drop=False, append=False, inplace=True)
            predictedDFs.append(predictedDF)
        
        outputDF = pd.concat(predictedDFs, axis=1)
        idx = outputDF.groupby(['patient'], sort=False)['predictionscore'].max() == outputDF['predictionscore']
        outputDF = outputDF[idx]
        outputDF.to_csv(outputfile, index=False, sep='\t')
    else:
        data = preprocessor.joinDatasetsToSingleDataset(datasets)
        X = data.getFullDataframe(False, False)
        predictions, scores = predictor.generate_predictions_scores(X, modelType)
        outputDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), predictions, scores)
        outputDF.set_index("patient", drop=False, append=False, inplace=True)
        outputDF.to_csv(outputfile, index=False, sep='\t')
    print("Sub Challenge 1 prediction finished...")


if __name__ == "__main__":
    main(sys.argv[1:])