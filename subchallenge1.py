#!/usr/bin/python

import sys, getopt, os

from machinelearning.vcf_model_predictor import VCFModelPredictor
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from preprocessor.vcf_features_selector import VCFFeaturesSelector


joinALLDatasets = True

def getReportByStudy(df):
    for study in df["study"].unique():
        studydf = df.loc[df['study'] == study]
        print("="*40)
        print("Results from study: " + str(study))
        prediction_report(studydf)

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
    print("="*40)
    print("Start using individual datasets to predict...")
    print("="*40)
    for modelType in datasets.keys():
        selector = VCFFeaturesSelector(datasets[modelType])
        dataset = selector.generateFilteredData()
        X = dataset.getFullDataframe(False, False)
        predictions, scores = predictor.generate_predictions_scores(X, modelType)
        predictedDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), modelType, predictions, scores)
        predictedDF.set_index("patient", drop=False, append=False, inplace=True)
        predictedDFs.append(predictedDF)
    
    print("Finished individual datasets prediction...")
    print("="*40)
    print("Start using global dataset to predict...")
    print("="*40)
    data = preprocessor.joinDatasetsToSingleDataset(datasets)
    allXDataset = data.getFullDataframe(False, False)
    predictionsAllXDataset, scoresAllXDataset = predictor.generate_predictions_scores(allXDataset, data.get_dataset_origin())
    predictedALLDF = predictor.generate_prediction_dataframe(preprocessor.getClinicalData(), data.get_dataset_origin(), predictionsAllXDataset, scoresAllXDataset)
    predictedALLDF.set_index("patient", drop=False, append=False, inplace=True)
    predictedDFs.append(predictedALLDF)
    print("Finished global dataset prediction...")
    print("="*40)
    return predictedDFs
    
def selectBestScoresFromDifferentModels(predictedDFs):
    outputDF = pd.concat(predictedDFs)
    outputDF = outputDF.sort_values('predictionscore', ascending=False).groupby('patient', as_index=False).first()
    return outputDF

def transformToRankingScore(x):
    if x['highriskflag'] == 'TRUE':
        return x['predictionscore']
    elif x['highriskflag'] == 'FALSE':
        return 1.0 - x['predictionscore']
    else:
        raise("FLAG NOT FOUND: " + x['highriskflag'])
    

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
    #datasetsUnfiltered = preprocessor.getPatientDataByDataset()
    datasetsFiltered = preprocessor.getPatientDataByDataset(useFiltered=True)
    print("Finished reading VCF Files...")
    
    predictor = VCFModelPredictor()
    predictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsFiltered)
    '''
    predictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsUnfiltered)
    filteredpredictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsFiltered)
    for predictedDF in filteredpredictedDFs:
        predictedDFs.append(predictedDF)
    '''
    outputDF = selectBestScoresFromDifferentModels(predictedDFs)
    print("="*40)
    print("Model fitment by study: ")
    getReportByStudy(outputDF)
    print("="*40)
    print("Model fitment scoring: ")
    prediction_report(outputDF)
    print("="*40)
    print("Model ranking scoring: ")
    outputDF['predictionscore'] = outputDF.apply(transformToRankingScore, axis=1)
    prediction_report(outputDF)
    print("="*40)
    outputDF.to_csv(outputfile, index=False, sep='\t')
    
    print("Sub Challenge 1 prediction finished...")


if __name__ == "__main__":
    main(sys.argv[1:])