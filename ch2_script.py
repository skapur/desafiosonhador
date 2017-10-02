#!/usr/bin/python

import pickle
import sys, getopt
from data_preprocessing import MMChallengeData, MMChallengePredictor
from genomic_data_test import df_reduce
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from copy import deepcopy

def prediction_report(df):
    # min, max, IQR, median, mean, Trues
    scores = df["predictionscore"]
    flags = df["highriskflag"]
    maxp, minp = scores.max(), scores.min()
    q1, q3 = scores.quantile([.25, .75])
    mean, median = scores.mean(), scores.median()
    num_trues = sum(flags == "TRUE")
    print("Score range: "+str(minp)+" <> "+str(maxp))
    print("Q1 = "+str(q1)+" <> "+"Q3 = "+str(q3))
    print("Mean: "+str(mean))
    print("Median: "+str(median))
    print("True predictions: "+str(num_trues))

def main(argv):
    #DIR = "/home/skapur/synapse/syn7222203/"
    #DIR = 'D:\Dropbox\workspace_intelij\DREAM_Challenge'
    #mmcd = MMChallengeData(DIR)
    print("Starting the script!")
    print("Reading clinical data")
    mmcd = MMChallengeData(sys.argv[1])

    print("Reading files using information from clinical data")

    with open('/desafiosonhador/colnames.sav','rb') as f:
        colname_dict = pickle.load(f)

    col_parse_dict = {
        ('RNASeq', 'trans'): lambda x: x.split('.')[0]
    }

    # mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable="D_Age", directoryFolder='/test-data/', columnNames=None, NARemove=[True,True], colParseFunDict=col_parse_dict)
    #
    # for key, df in mmcd.dataDict.items():
    #     print(key)
    #     print("First 100 features - Validation: "+str(df[0].columns.tolist()[:100]))
    #     overlap = set(colname_dict[key]) & set(df[0].columns.tolist())
    #     print("Overlapped genes:")
    #     print(overlap)
    #     print("Dataframe columns: "+str(df[0].shape[1]))
    #     print("Amount of full NA columns: "+str(df[0].isnull().all().sum()))
    #     print("Amount of partial NA columns: " + str(df[0].isnull().any().sum()))
    #     print("*"*80)


    mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable="D_Age", directoryFolder='/test-data/', columnNames=colname_dict, NARemove=[True,True], colParseFunDict=col_parse_dict)

    for key, df in mmcd.dataDict.items():
        print(key)
        #print("First 100 features - Validation: "+str(df[0].columns.tolist

        training_features = set(colname_dict[key])
        validation_features = set(df[0].columns.tolist())
        overlapped_features = training_features & validation_features

        print(str(len(overlapped_features))+" "+"overlapped features.")
        #print("Dataframe columns: "+str(df[0].shape[1]))
        #print("Amount of full NA columns: "+str(df[0].isnull().all().sum()))
        #print("Amount of partial NA columns: " + str(df[0].isnull().any().sum()))
        print("*"*80)

    # ======== RNA-SEQ ========

    print("Loading RS transformer")
    #Load fitted transformers and model
    with open('/desafiosonhador/transformers_rna_seq.sav', 'rb') as f:
        trf_rseq = pickle.load(f)

    print("Loading RS classifier")
    with open('/desafiosonhador/fittedModel_rna_seq.sav', 'rb') as f:
        clf_rseq = pickle.load(f)

    # Redefining scaler for RNA-Seq
    rseq_new_scl = MaxAbsScaler()
    rseq_data = pd.concat([mmcd.dataDict[("RNASeq", "gene")][0], mmcd.dataDict[("RNASeq", "trans")][0]], axis=1)
    rseq_new_scl.fit(rseq_data)
    trf_rseq['fts'] = rseq_new_scl


    mv_fun_rseq = lambda x: df_reduce(x.values.reshape(1,-1), [], fit = False, scaler = trf_rseq['scaler'], fts = trf_rseq['fts'])[0]

    print("Predicting with RS...")
    # Make predictions
    mod_rseq = MMChallengePredictor(
            mmcdata = mmcd,
            predict_fun = lambda x: clf_rseq.predict(x)[0],
            confidence_fun = lambda x: clf_rseq.predict_proba(x)[0][1],
            data_types = [("RNASeq", "gene"), ("RNASeq", "trans")],
            single_vector_apply_fun = lambda x: x,
            multiple_vector_apply_fun = mv_fun_rseq
    )
    res_rseq = mod_rseq.predict_dataset()


    # ======== MICROARRAYS ========
    print("Loading MA transformer")
    #Load fitted transformers and model
    with open('/desafiosonhador/transformers_microarrays.sav', 'rb') as f:
        trf_marrays = pickle.load(f)

    print("Loading MA classifier")
    with open('/desafiosonhador/fittedModel_microarrays.sav', 'rb') as f:
        clf_marrays = pickle.load(f)

    # Redefining scaler for marrays
    marrays_new_scl = MaxAbsScaler()
    marrays_data = mmcd.dataDict[("MA", "gene")][0]
    marrays_new_scl.fit(marrays_data)
    trf_marrays['fts'] = marrays_new_scl

    mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = trf_marrays['scaler'], fts = trf_marrays['fts'], fit = False)[0]

    # Make predictions
    print("Predicting with MA...")
    mod_marryas = MMChallengePredictor(
                mmcdata = mmcd,
                predict_fun = lambda x: clf_marrays.predict(x)[0],
                confidence_fun = lambda x: clf_marrays.predict_proba(x)[0][1],
                data_types = [("MA", "gene")],
                single_vector_apply_fun = mv_fun,
                multiple_vector_apply_fun = lambda x: x
    )
    res_marrays = mod_marryas.predict_dataset()

    #Final dataset
    print("Generating prediction matrix")
    final_res = res_rseq.combine_first(res_marrays)
    final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
    final_res['highriskflag'] = final_res['highriskflag'] == 1
    final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())

    print("Writing prediction matrix")
    final_res.to_csv(sys.argv[2], index = False, sep = '\t')

    print("Any failed prediction column in the prediction matrix?")
    print(str(final_res.isnull().any()))
    print("All failed prediction column in the prediction matrix?")
    print(str(final_res.isnull().all()))

    final_res["predictionscore"] = final_res["predictionscore"].fillna(value=0)
    final_res["highriskflag"] = final_res["highriskflag"].fillna(value=False)

    prediction_report(final_res)
    print("Done!")

if __name__ == "__main__":
    main(sys.argv[1:])