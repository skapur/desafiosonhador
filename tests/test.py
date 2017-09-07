import sys
sys.path.insert(0,'/home/dreamchallenge/python_scripts/desafiosonhador')
from data_preprocessing import MMChallengeData
import os.path as path
import pandas as pd
from pandas.core.frame import DataFrame
if __name__ == '__main__':
    
    patha = '/home/dreamchallenge/synapse/syn7222203';
    mmcd = MMChallengeData(patha)
    df = mmcd.getDataFrame("Genomic", "StrelkaIndels", savesubdataframe='/home/dreamchallenge/synapse/syn7222203/StrelkaIndels.csv')
    df.to_csv("/home/dreamchallenge/synapse/syn7222203/StrelkaIndels_joined.csv")
    
    '''
    df = DataFrame.from_csv('/home/tiagoalves/rrodrigues/MuTectsnvs.csv')
    df = df.T
    df = df[:10]
    df.set_index(df.columns[0], drop=False, append=False, inplace=True)
    print(df)
    df2 = DataFrame.from_csv('/home/tiagoalves/rrodrigues/globalClinTraining.csv')
    df2.set_index(df2.columns[18], drop=False, append=True, inplace=True)
    df3 = pd.concat([df, df2], axis=1)
    df3.to_csv('/home/tiagoalves/rrodrigues/test.csv')
    '''