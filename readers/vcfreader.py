import multiprocessing
import vcf
import pandas as pd
import pickle


class VCFReader(object):

    def __init__(self):
        self.__vcfinstances = {}
        self.__genes = set()
        self.__genesFunct = set()
        self.__hasheffect = {'HIGH':10000, 'MODERATE':500, 'MODIFIER':30, 'LOW':1}
        f = open("serialized_features/filteringGenesAndFunctions.pkl", "rb")
        self.__filtering = pickle.load(f)
        f.close()
    
    def readVCFFileFindCompression(self, filename):
        compressed = filename.endswith(".gz")
        return self.readVCFFile(filename, compressed)
        
    
    def readVCFFile(self, filename, compressed=True):
        genetoscore = {}
        gene_function = {}
        genes_tlod = {}
        genes_bigqss = {}
        genes_qss = {}
        genes_clustered = {}
        genes_germline_risk = {}
        genes_somatic_risk = {}
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                ann = record.INFO['ANN']
                memorized_geneinstances = set()
                for firstann in ann:
                    firstann = firstann.split('|')
                    gene_instance = firstann[3]
                    score_instance = self.__hasheffect[firstann[2]]
                    functionAnnotation = firstann[1]
                    if gene_instance and gene_instance not in memorized_geneinstances:
                        memorized_geneinstances.add(gene_instance)
                        self.__genes.add(gene_instance)
                        if functionAnnotation in self.__filtering["functions"] and gene_instance in self.__filtering["genes"]:
                            gene_function[gene_instance+"_"+functionAnnotation] = 1
                        if gene_instance in genetoscore:
                            genetoscore[gene_instance] = genetoscore[gene_instance] + score_instance
                        else:
                            genetoscore[gene_instance] = score_instance
                    if 'TLOD' in record.INFO.keys() and 'NLOD' in record.INFO.keys():
                        tlod = record.INFO['TLOD']
                        nlod = record.INFO['NLOD']
                        if tlod > nlod:
                            genes_tlod["TLOD_"+gene_instance] = 1
                    
                    if 'QSS' in record.INFO.keys() and 'QSS_NT' in record.INFO.keys():
                        qss = record.INFO['QSS']
                        qss_nt = record.INFO['QSS_NT']
                        if qss > 10:
                            genes_qss["QSS_"+gene_instance] = 1
                        if qss > qss_nt:
                            genes_bigqss["BIGQSS_"+gene_instance] = 1
                    if 'ECNT' in record.INFO.keys():
                        ecnt = record.INFO['ECNT']
                        if float(ecnt) > 1: 
                            genes_clustered["Clustered_"+gene_instance] = 1
                    if 'HCNT' in record.INFO.keys():
                        hcnt = record.INFO['HCNT']
                        if float(hcnt) > 1:
                            genes_clustered["Clustered_"+gene_instance] = 1
                    if 'SAO' in record.INFO.keys():
                        sao = record.INFO['SAO']
                        if sao == 1 or sao == 3:
                            genes_germline_risk["Germline_"+gene_instance] = 1
                        if sao == 2 or sao == 3:
                            genes_somatic_risk["Somatic_"+gene_instance] = 1
                    
            #else:
                #print(record)
                #print('File: '+filename+" don't have ANN entry")
        return genetoscore, gene_function, genes_tlod, genes_qss, genes_bigqss, genes_clustered, genes_germline_risk, genes_somatic_risk
    
    def getAllFunctions(self, filenames):
        functionAnnotations = set()
        executor = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        for v in executor.map(self.getFunctions, filenames):
            for z in v:
                functionAnnotations.add(z)
        
        print(functionAnnotations)
    
    def getAllFunctionsWithTrueAndFalses(self, filenames, listLabels, outputFile):
        truefunctionAnnotations = set()
        falsefunctionAnnotations = set()
        executor = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        for l, v in zip(listLabels, executor.map(self.getFunctions, filenames)):
            if l == "TRUE":
                for z in v:
                    truefunctionAnnotations.add(z)
            elif l == "FALSE":
                for z in v:
                    falsefunctionAnnotations.add(z)
        
        
        #outputFile.write("True functions\n")
        #outputFile.write(str(truefunctionAnnotations)+"\n")
        #outputFile.write("False functions"+"\n")
        #outputFile.write(str(falsefunctionAnnotations)+"\n")
        intersectionAnnotations = truefunctionAnnotations.intersection(falsefunctionAnnotations)
        outputFile.write("only in true functions"+"\n")
        outputFile.write(str(truefunctionAnnotations - intersectionAnnotations) +"\n")
        outputFile.write("only in false functions"+"\n")
        outputFile.write(str(falsefunctionAnnotations - intersectionAnnotations) +"\n")
    
    def getFunctions(self, filename, compressed=True):
        functionAnnotations = set()
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                ann = record.INFO['ANN']
                for firstann in ann:
                    firstann = firstann.split('|')
                    if firstann[3] and firstann[1]:
                        functionAnnotations.add(firstann[3]+"_"+firstann[1])
        return functionAnnotations
    
    def getGenesWithUpperTLOD(self, filename, compressed=True):
        genes = set()
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                ann = record.INFO['ANN']
                for firstann in ann:
                    firstann = firstann.split('|')
                    gene_instance = firstann[3]
                    if 'TLOD' in record.INFO.keys() and 'NLOD' in record.INFO.keys():
                        tlod = record.INFO['TLOD']
                        nlod = record.INFO['NLOD']
                        if tlod > nlod:
                            genes.add("TLOD_"+gene_instance)
        return genes
    
    def getGenesWithUpperQSI(self, filename, compressed=True):
        genes = set()
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                ann = record.INFO['ANN']
                for firstann in ann:
                    firstann = firstann.split('|')
                    gene_instance = firstann[3]
                    if 'QSS' in record.INFO.keys() and 'QSS_NT' in record.INFO.keys():
                        qss = record.INFO['QSS']
                        qss_nt = record.INFO['QSS_NT']
                        if qss > 10:
                            genes.add("QSS_"+gene_instance)
                        if qss > qss_nt:
                            genes.add("BIG_QSS_"+gene_instance)
        return genes
    
    def getGenesWithVlustered(self, filename, compressed=True):
        genes = set()
        vcfrecords = vcf.Reader(filename=filename, compressed=compressed)
        for record in vcfrecords:
            if 'ANN' in record.INFO.keys():
                ann = record.INFO['ANN']
                for firstann in ann:
                    firstann = firstann.split('|')
                    gene_instance = firstann[3]
                    if 'clustered_events' in record.FILTER:
                        genes.add(gene_instance)

        return genes

    
    def readFiles(self, files):
        for filepath in files:
            self.__vcfinstances[filepath] = self.readVCFFile(filename=filepath, compressed=True)
    
    def getDataframe(self):
        dataframe = pd.DataFrame(data=self.vcfinstances)
        dataframe.fillna(value=0, inplace=True)
        return dataframe