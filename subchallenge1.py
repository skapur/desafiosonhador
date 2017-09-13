#!/usr/bin/python

import sys, getopt
import data_preprocessing as processor
import pandas as pd
import pickle

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
    
    processingData = processor.MMChallengeData(inputfile);
    x, y, modelType = processingData.preprocessPrediction(outputVariable="D_Age");
    
    f = open(trained_Models[modelType]["__columnsDic"], 'rb')
    featColumns = pickle.load(f);
    f.close();
    
    f = open(trained_Models[modelType]["__classifierFilename"], 'rb')
    clf = pickle.load(f)
    f.close();

    x = x.loc[:, featColumns]
    x = x.fillna(value=0)

    x, y, z = processor.df_reduce(x, y, fit=False, filename=trained_Models[modelType]["__transformerFilename"])
    
    predictions = clf.predict(x)
    scores = clf.predict_proba(x)[:,1]

    predicted = pd.DataFrame({"predictionscore":scores, "highriskflag":predictions}, index=processingData.clinicalData.index)
    
    information = processingData.clinicalData[["Study","Patient"]]
    outputDF = pd.concat([information, predicted], axis=1)
    outputDF = outputDF[["Study","Patient", "predictionscore", "highriskflag"]]
    outputDF.columns = ["study","patient", "predictionscore", "highriskflag"]
    outputDF.to_csv(outputfile, index = False)
    
    '''
    my_fun = lambda x: processor.df_reduce(x.values.reshape(1, -1), [], fit=False, filename=trained_Models[modelType]["__transformerFilename"])[0]
    
    processingData.dataDict = {"genomic" : (x,[],y) }
    processingData.generateDataTypePresence()

    mod = processor.MMChallengePredictor(
        mmcdata = processingData,
        predict_fun = lambda x: clf.predict(x)[0],
        confidence_fun = lambda x: 1 - min(clf.predict_proba(x)[0]),
        data_types = ["genomic"],
        single_vector_apply_fun = my_fun,
        multiple_vector_apply_fun = lambda x: x)
    
    outputDF = mod.predict_dataset()
    outputDF.to_csv(outputfile, index = False)
    '''

if __name__ == "__main__":
    main(sys.argv[1:])