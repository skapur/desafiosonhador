from data_preprocessing import MMChallengeData
import os.path as path
import pandas as pd
from pandas.core.frame import DataFrame
if __name__ == '__main__':
    
    patha = 'C:/Users/ru13e/Desktop/LxMLS/vcf_reader/syn7222203';

    mmcd = MMChallengeData(patha)
    '''
    df = mmcd.getDataFrame("Genomic", "MuTectsnvs")
    df.to_csv("C:/Users/ru13e/Desktop/LxMLS/vcf_reader/test.csv")
    '''
    df = DataFrame.from_csv(path.join(patha, "MuTectsnvs.csv"))
    df = df.T
    df.columns.values[0] = "file" 
    print(df)
    df2 = mmcd.clinicalData[["Patient", "WES_mutationFileMutect"]] 
    df3 = pd.concat([df, df2], keys=["WES_mutationFileMutect","file"])
    df3.to_csv("C:/Users/ru13e/Desktop/LxMLS/vcf_reader/test.csv")
