#import vcf
import pandas as pd

class VCFReader(object):

    def __init__(self):
        self.__vcfinstances = {}
        self.__genes = set()
        self.__hasheffect = {'HIGH':10, 'MODERATE':5, 'MODIFIER':3, 'LOW':1}
    
    def readVCFFile(self, filename, compressed=True):
        genetoscore = {}
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                firstann = record.INFO['ANN'][0]
                firstann = firstann.split('|')
                gene_instance = firstann[3]
                score_instance = self.__hasheffect[firstann[2]]
                if gene_instance:
                    self.__genes.add(gene_instance)
                    if gene_instance in genetoscore:
                        genetoscore[gene_instance] = genetoscore[gene_instance] + score_instance
                    else:
                        genetoscore[gene_instance] = score_instance
            #else:
                #print(record)
                #print('File: '+filename+" don't have ANN entry")
        return genetoscore
    
    def readFiles(self, files):
        for filepath in files:
            self.__vcfinstances[filepath] = self.readVCFFile(filename=filepath, compressed=True)
    
    def getDataframe(self):
        dataframe = pd.DataFrame(data=self.vcfinstances)
        dataframe.fillna(value=0, inplace=True)
        return dataframe