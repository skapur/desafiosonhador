import pickle
import sys
import numpy as np
import pandas as pd

from data_preprocessing import MMChallengeData, MMChallengePredictor
from sklearn.preprocessing import Imputer


def prediction_report(df, confidence=False):
    # min, max, IQR, median, mean, Trues
    scores = df["predictionscore"].copy()
    flags = df["highriskflag"].copy()
    if confidence:
        scores = (scores - 0.5).abs() + 0.5
    maxp, minp = scores.max(), scores.min()
    q1, q3 = scores.quantile([.25, .75])
    mean, median = scores.mean(), scores.median()
    num_trues = sum(flags == "TRUE")
    print("Score range: "+str(minp)+" <> "+str(maxp))
    print("Q1 = "+str(q1)+" <> "+"Q3 = "+str(q3))
    print("Mean: "+str(mean))
    print("Median: "+str(median))
    print("True predictions: "+str(num_trues))

def save_as_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def data_dict_report(data_dict, colname_dict):
    for key, df in data_dict.items():
        try:
            print(key)
            training_features = set(colname_dict[key])
            validation_features = set(df[0].columns.tolist())
            overlapped_features = training_features & validation_features

            print(str(len(overlapped_features)) + " " + "overlapped features.")
            print("Dataframe has " + str(len(df[0].columns.tolist())))
            print("Dataframe columns: " + str(df[0].shape[1]))
            print("Full NA columns: " + str(df[0].isnull().all().sum() > 0))
            print("Partial NA columns: " + str(df[0].isnull().any().sum() > 0))
            print("*" * 80)
        except:
            pass

def binarize_rows_by_quantile(df, q):
    return (df.T > df.quantile(q, axis=1)).T

def generate_binary_features(x, qs):
    return pd.concat([binarize_rows_by_quantile(x, q) for q in qs], axis=1).astype(int)

def minmax(x):
    return (x-min(x))/(max(x)-min(x))


def main(argv):
    print("Starting the script!")
    clin_file_path = '/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv'
    data_file_path = '/home/skapur/synapse/syn7222203/CH2'
    # clin_file_path = sys.argv[1]
    # data_file_path = '/test-data/'

    print("Reading clinical data")
    mmcd = MMChallengeData(clin_file_path)

    print('Reading feature list')
#    colname_dict = read_pickle('/desafiosonhador/colnames_r3.sav')
    colname_dict = {
        ('RNASeq','gene'): read_pickle('RNASeq_genes_08112017'),
        ('MA','gene'): read_pickle('MA_genes_08112017'),
        ('MA','probe'):[],
        ('RNASeq','trans'):[]
    }

    print('Defining clinical variables')
    col_parse_dict = {
        ('RNASeq', 'trans'): lambda x: x.split('.')[0]
    }
    clinical_variables = ["D_Age", "D_ISS"]
    output_variable = "D_Age" # Placeholder as it's not being used
    mmcd.clinicalData[['D_Age','D_ISS']]
    print('Generating data dictionary')
    mmcd.generateDataDict(clinicalVariables=clinical_variables,
                          outputVariable=output_variable,
                          directoryFolder=data_file_path,
                          columnNames=colname_dict,
                          NARemove=[False, True],
                          colParseFunDict=col_parse_dict
    )
    # x - dataframe; qs - quantile list
    # Generate new RNA_Seq features

    print('Processing RNA-Seq data...')
    RNA_quantile_steps = np.linspace(0.1, 0.9, 5)
    RNA_X, RNA_C, RNA_y = mmcd.dataDict[('RNASeq','gene')]
    RNA_C = RNA_C.T.drop_duplicates().T
    RNA_Xb = generate_binary_features(RNA_X, RNA_quantile_steps)
    print('Engineering RNA-Seq features')

    #print('Adding RNA-Seq features to data_dict')
    #mmcd.addToDataDict(('RNASeq','binary_gene'), RNA_Xb, RNA_C, RNA_y)

    data_dict_report(mmcd.dataDict, colname_dict)


    ########################
    ##
    # RNA-SEQ CLASSIFIERS
    ##
    ########################
    # RNA_transformer = read_pickle('desafiosonhador/rnaseq_stack_pipeline_08112017')
    # RNA_classifier = read_pickle('desafiosonhador/rnaseq_stack_classifier_08112017')
    print('Reading serialized RNA-Seq data classifiers/transformers')
    RNA_transformer = read_pickle('rnaseq_stack_pipeline_08112017')
    RNA_classifier = read_pickle('rnaseq_stack_classifier_08112017')

    print('Transforming RNA-Seq data')
    RNA_imputer = Imputer(strategy='median', axis=0)

    RNA_Xbd_imp = RNA_imputer.fit_transform(RNA_Xb)
    RNA_Xbd_sel = RNA_transformer.transform(RNA_Xbd_imp)

    RNA_C = RNA_C.apply(minmax)
    RNA_C['D_AgeISSMean'] = RNA_C.mean(axis=1)
    RNA_C_imp = pd.DataFrame(RNA_imputer.fit_transform(RNA_C), index=RNA_C.index)
    RNA_x_final = pd.DataFrame(np.concatenate([RNA_Xbd_sel, RNA_C_imp], axis=1), index=RNA_X.index)

    print('Adding transformed RNA-Seq data to data dictionary')
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
    #RNA_prediction_df.to_csv('RNA_prediction_df.test.csv')

    ########################
    ##
    # MICROARRAY CLASSIFIERS
    ##
    ########################

    MA_transformer = read_pickle('transformers_microarrays.sav')
    MA_classifier = read_pickle('ma_voting_clf.sav')
    from genomic_data_test import df_reduce

    mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = MA_transformer['scaler'], fts = MA_transformer['fts'], fit = False)[0]

    # Make predictions
    # proba_fun_ma = lambda x: np.array([clf.predict_proba(x)[0][1] for name, clf in clf_list_marrays]).reshape(1, -1)
    # pred_fun_ma = lambda x: clf_marrays.predict(proba_fun_ma(x))[0]
    # conf_fun_ma = lambda x: clf_marrays.predict_proba(proba_fun_ma(x))[0][1]

    print("Predicting with MA...")
    MA_predictor = MMChallengePredictor(
                mmcdata = mmcd,
                predict_fun = lambda x: MA_classifier.predict(x)[0],
                confidence_fun = lambda x: MA_classifier.predict_proba(x)[0][1],
                data_types = [("MA", "gene")],
                single_vector_apply_fun = mv_fun,
                multiple_vector_apply_fun = lambda x: x
    )
    MA_prediction_df = MA_predictor.predict_dataset()

    print("Generating prediction matrix")
    final_res = MA_prediction_df.combine_first(RNA_prediction_df)
    final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
    final_res['highriskflag'] = final_res['highriskflag'] == 1
    final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())

    print("Any failed prediction column in the prediction matrix?")
    print(str(final_res.isnull().sum()))

    final_res["predictionscore"] = final_res["predictionscore"].fillna(value=0)
    final_res["highriskflag"] = final_res["highriskflag"].fillna(value=False)

    print("Writing prediction matrix")
    final_res.to_csv(sys.argv[2], index = False, sep = '\t')
    print('*'*80)

    def print_confidence(df):
        print(df['study'].unique())
        print(df.shape)
        prediction_report(df, True)
        print('*' * 80)

    final_res.groupby('study').apply(print_confidence)

    print('Model fitting scores:')
    prediction_report(final_res, True)
    print('*'*80)
    print('Model score distribution')
    prediction_report(final_res, False)

    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])

