import pickle

from sklearn.externals.joblib.numpy_pickle_utils import np
from sklearn.preprocessing.imputation import Imputer

from ch2_script import read_pickle, generate_binary_features, data_dict_report, \
    minmax
from ch2_training_resources import df_reduce
from data_preprocessing import MMChallengeData, MMChallengePredictor
from machinelearning.vcf_model_predictor import VCFModelPredictor
import pandas as pd
from preprocessor.vcf_data_preprocessing import VCFDataPreprocessor
from subchallenge1 import generateSubModelPredictions, \
    selectBestScoresFromDifferentModels, transformToRankingScore
from trainvcfmodel import read_serialized_dataset


def processUsingCH1Method(inputfile):
    preprocessor = VCFDataPreprocessor(inputfile)
    #datasetsUnfiltered = preprocessor.getPatientDataByDataset()
    datasetsFiltered = preprocessor.getPatientDataByDataset(useFiltered=True)
    print("Finished reading VCF Files...")
    
    predictor = VCFModelPredictor()
    predictedDFs = generateSubModelPredictions(preprocessor, predictor, datasetsFiltered)
    outputDF = selectBestScoresFromDifferentModels(predictedDFs)
    outputDF['predictionscore'] = outputDF.apply(transformToRankingScore, axis=1)
    return outputDF

def processUsingCH2Method(inputfile):
    mmcd = MMChallengeData(inputfile)
    colname_dict = {
        ('RNASeq','gene'): read_pickle('/desafiosonhador/RNASeq_genes_08112017'),
        ('MA','gene'): read_pickle('/desafiosonhador/MA_genes_08112017'),
        ('MA','probe'):[],
        ('RNASeq','trans'):[]
    }

    col_parse_dict = {
        ('RNASeq', 'trans'): lambda x: x.split('.')[0]
    }
    clinical_variables = ["D_Age", "D_ISS"]
    output_variable = "D_Age" # Placeholder as it's not being used
    mmcd.clinicalData[['D_Age','D_ISS']]

    mmcd.generateDataDict(clinicalVariables=clinical_variables,
                          outputVariable=output_variable,
                          directoryFolder='/test-data/',
                          columnNames=colname_dict,
                          NARemove=[False, True],
                          colParseFunDict=col_parse_dict
    )

    RNA_quantile_steps = np.linspace(0.1, 0.9, 5)
    RNA_X, RNA_C, RNA_y = mmcd.dataDict[('RNASeq','gene')]
    RNA_C = RNA_C.T.drop_duplicates().T
    RNA_Xb = generate_binary_features(RNA_X, RNA_quantile_steps)

    data_dict_report(mmcd.dataDict, colname_dict)

    RNA_transformer = read_pickle('/desafiosonhador/rnaseq_stack_pipeline_08112017')
    RNA_classifier = read_pickle('/desafiosonhador/rnaseq_stack_classifier_08112017')

    RNA_imputer = Imputer(strategy='median', axis=0)

    RNA_Xbd_imp = RNA_imputer.fit_transform(RNA_Xb)
    RNA_Xbd_sel = RNA_transformer.transform(RNA_Xbd_imp)

    RNA_C = RNA_C.apply(minmax)
    RNA_C['D_AgeISSMean'] = RNA_C.mean(axis=1)
    RNA_C_imp = pd.DataFrame(RNA_imputer.fit_transform(RNA_C), index=RNA_C.index)
    RNA_x_final = pd.DataFrame(np.concatenate([RNA_Xbd_sel, RNA_C_imp], axis=1), index=RNA_X.index)

    mmcd.addToDataDict('RNASeq','binary_feature_selected', RNA_x_final, RNA_C_imp, RNA_y)

    RNA_predictor = MMChallengePredictor(
            mmcdata = mmcd,
            predict_fun = lambda x: RNA_classifier.predict(x)[0],
            confidence_fun = lambda x: RNA_classifier.predict_proba(x)[0][1],
            data_types = [("RNASeq", "binary_feature_selected")],
            single_vector_apply_fun = lambda x: x.values.reshape(1,-1),
            multiple_vector_apply_fun = lambda x: x
    )

    RNA_prediction_df = RNA_predictor.predict_dataset()

    MA_transformer = read_pickle('/desafiosonhador/transformers_microarrays.sav')
    MA_classifier = read_pickle('/desafiosonhador/ma_voting_clf.sav')
    from genomic_data_test import df_reduce

    mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = MA_transformer['scaler'], fts = MA_transformer['fts'], fit = False)[0]

    MA_predictor = MMChallengePredictor(
                mmcdata = mmcd,
                predict_fun = lambda x: MA_classifier.predict(x)[0],
                confidence_fun = lambda x: MA_classifier.predict_proba(x)[0][1],
                data_types = [("MA", "gene")],
                single_vector_apply_fun = mv_fun,
                multiple_vector_apply_fun = lambda x: x
    )
    MA_prediction_df = MA_predictor.predict_dataset()

    final_res = MA_prediction_df.combine_first(RNA_prediction_df)
    final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
    final_res['highriskflag'] = final_res['highriskflag'] == 1
    final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())
    final_res["predictionscore"] = final_res["predictionscore"].fillna(value=0)
    final_res["highriskflag"] = final_res["highriskflag"].fillna(value=False)

    return final_res, mmcd.clinicalData[["D_Age", "D_ISS"]]

def joinAndPredictDataframes(inputfile):
    out1 = processUsingCH1Method(inputfile)
    out1.index = out1['patient']
    out2, clinical = processUsingCH2Method(inputfile)
    out2.index = out2['patient']
    
    datainfo = pd.DataFrame(pd.concat([clinical, out1["predictionscore"], out2["predictionscore"]],axis=1), columns=['D_Age', 'D_ISS','challenge1_scores', 'challenge2_scores'])


    return datainfo, out1, out2

def generateSerializedScoresChallenge1():
    paths = ['/home/tiagoalves/rrodrigues/vcf-datasets_v5/MuTectsnvs_filtered_dataset_CH1.pkl',
             #'/home/tiagoalves/rrodrigues/vcf-datasets/StrelkaIndels_dataset_CH1.pkl',
             '/home/tiagoalves/rrodrigues/vcf-datasets_v5/Strelkasnvs_filtered_dataset_CH1.pkl']
    datasets = {}
    preprocessor = VCFDataPreprocessor(None)
    for pat in paths:
        dataset = read_serialized_dataset(pat)
        datasets[dataset.get_dataset_origin()] = dataset
    predictor = VCFModelPredictor()
    data = preprocessor.prepareDatasetForStacking(datasets)
    allXDataset = data.getFullDataframe(False, False)
    predictionsAllXDataset, scoresAllXDataset = predictor.generate_predictions_scores(allXDataset, data.get_dataset_origin())
    predictedALLDF = predictor.generate_prediction_dataframe_serial(data.get_patients(), data.get_dataset_origin(), predictionsAllXDataset, scoresAllXDataset)
    predictedALLDF.set_index("patient", drop=False, append=False, inplace=True)
    predictedALLDF.to_csv("/home/tiagoalves/rrodrigues/desafiosonhador/predictions_ch1.csv")

if __name__ == '__main__':
    generateSerializedScoresChallenge1()