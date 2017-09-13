#!/usr/bin/python

import pickle
import sys, getopt
from data_preprocessing import MMChallengeData, MMChallengePredictor
from genomic_data_test import df_reduce

def main(argv):
    #DIR = "/home/skapur/synapse/syn7222203/"
    #DIR = 'D:\Dropbox\workspace_intelij\DREAM_Challenge'
    #mmcd = MMChallengeData(DIR)
    print("Starting the script!")
    print("Reading clinical data")
    mmcd = MMChallengeData(sys.argv[1])
    print("Reading files using information from clinical data")
    mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable=["D_Age"], directoryFolder='/test-data/')

    # ======== RNA-SEQ ========

    print("Loading RS transformer")
    #Load fitted transformers and model
    with open('/desafiosonhador/transformers_rna_seq.sav', 'rb') as f:
        trf_rseq = pickle.load(f)

    print("Loading RS classifier")
    with open('/desafiosonhador/fittedModel_rna_seq.sav', 'rb') as f:
        clf_rseq = pickle.load(f)

    mv_fun_rseq = lambda x: df_reduce(x.values.reshape(1,-1), [], fit = False, scaler = trf_rseq['scaler'], fts = trf_rseq['fts'])[0]

    print("Predicting with RS...")
    # Make predictions
    mod_rseq = MMChallengePredictor(
            mmcdata = mmcd,
            predict_fun = lambda x: clf_rseq.predict(x)[0],
            # confidence_fun = lambda x: 1 - min(clf_rseq.predict_proba(x)[0]),
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

    print("Any failed prediction in the prediction matrix? "+str(final_res.isnull().any()))
    print("Done!")
if __name__ == "__main__":
    main(sys.argv[1:])