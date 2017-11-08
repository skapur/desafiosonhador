#!/usr/bin/python

import sys, getopt, os
import pandas as pd
from challengesJoiner import joinAndPredictDataframes
from ch2_script import read_pickle
from sklearn.preprocessing import Imputer


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
            
    df, out1, out2 = joinAndPredictDataframes(inputfile)
    metaclf = read_pickle('ch3_logreg_meta')

    patientmap = pd.concat([out1[['study','patient']], out2[['study','patient']]], axis=0).drop_duplicates()
    patientmap.index = patientmap['patient']

    df_drop = df.drop(df[df.isnull()['challenge1_scores']].index, axis=0).drop(df[df.isnull()['challenge2_scores']].index, axis=0)
    df_imp = Imputer(strategy='median', axis=0).fit_transform(df_drop)

    df_drop_score_df = patientmap.loc[df_drop.index,:]
    df_both = metaclf.predict(df_imp)
    df_both_proba = metaclf.predict_proba(df_imp)[:,1]
    df_drop_score_df['predictionscore'] = df_both_proba
    df_drop_score_df['highriskflag'] = df_both
    df_drop_score_df['highriskflag'] = df_drop_score_df['highriskflag'].astype(bool).astype(str).apply(lambda x: x.upper())


    df_ndrop = df.drop(df_drop.index, axis=0)
    ch1df = out1.loc[df_ndrop['CH1'].isnotnull().index,:]
    ch2df = out2.loc[df_ndrop['CH2'].isnotnull().index,:]

    final_pred = pd.concat([df_drop_score_df, ch1df, ch2df])
    final_pred.to_csv(outputfile, sep='\t', index = False)


        