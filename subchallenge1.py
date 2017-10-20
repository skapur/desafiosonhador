#!/usr/bin/python

import sys, getopt, os

from machinelearning.vcf_model_predictor import VCFModelPredictor
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from preprocessor.vcf_features_selector import VCFFeaturesSelector


joinALLDatasets = True

def prediction_report(df):
    # min, max, IQR, median, mean, Trues
    scores = df["predictionscore"]
    flags = df["highriskflag"]
    maxp, minp = scores.max(), scores.min()
    q1, q3 = scores.quantile([.25, .75])
    mean, median = scores.mean(), scores.median()
    num_trues = sum(flags == "TRUE")
    print("Dataset size: " + str(len(scores)))
    print("Score range: "+str(minp)+" <> "+str(maxp))
    print("Q1 = "+str(q1)+" <> "+"Q3 = "+str(q3))
    print("Mean: "+str(mean))
    print("Median: "+str(median))
    print("True predictions: "+str(num_trues))

def generateSubModelPredictions(preprocessor, predictor, datasets):
    predictedDFs = []
    for modelType in datasets.keys():
        selector = VCFFeaturesSelector(datasets[modelType])
        dataset = selector.generateFilteredData()
        X = dataset.getFullDataframe(False, False)
        predictions, scores = predictor.generate_predictions_scores(X, modelType)
        predictedDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), modelType, predictions, scores)
        predictedDF.set_index("patient", drop=False, append=False, inplace=True)
        predictedDFs.append(predictedDF)
    data = preprocessor.joinDatasetsToSingleDataset(datasets)
    predictions, scores = predictor.generate_predictions_scores(X, data.get_dataset_origin())
    predictedDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), data.get_dataset_origin(), predictions, scores)
    predictedDF.set_index("patient", drop=False, append=False, inplace=True)
    predictedDFs.append(predictedDF)
    return predictedDFs
    
def selectBestScoresFromDifferentModels(predictedDFs):
    outputDF = pd.concat(predictedDFs)
    outputDF = outputDF.sort_values('predictionscore', ascending=False).groupby('patient', as_index=False).first()
    return outputDF

def main(argv):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
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
    datasetsUnfiltered = preprocessor.getPatientDataByDataset()
    datasetsFiltered = preprocessor.getPatientDataByDataset(useFiltered=True)
    print("Finished reading VCF Files...")
    
    predictor = VCFModelPredictor()
    predictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsUnfiltered)
    filteredpredictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsFiltered)
    for predictedDF in filteredpredictedDFs:
        predictedDFs.append(predictedDF)
    
    outputDF = selectBestScoresFromDifferentModels(predictedDFs)
    outputDF.to_csv(outputfile, index=False, sep='\t')
    prediction_report(outputDF)
    print("Sub Challenge 1 prediction finished...")


if __name__ == "__main__":
    main(sys.argv[1:])