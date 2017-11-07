import pandas as pd
class PatientData(object):
    
    def __init__(self, dataset_origin, patients):
        '''
        Each init variable will contain a dataframe of patient - name of var.
        This object will allow the separation of patients from different datasets
        '''
        self.__dataset_origin = dataset_origin
        if not isinstance(patients, pd.Series):
            raise Exception("Patients must be a Series")
        self.__patients = patients
        self.__ages = None
        self.__ageRisk = None
        self.__ISSs = None
        self.__genes_scoring = None
        self.__genes_function_associated = None
        self.__genes_tlod = None
        self.__genes_qss = None
        self.__genes_big_qss = None
        self.__cytogenetic_features = None
        self.__flags = None

    def get_dataset_origin(self):
        return self.__dataset_origin
    
    def get_patients(self):
        return self.__patients
    
    def get_ages(self):
        return self.__ages
    
    def get_ageRisk(self):
        return self.__ageRisk
    
    def get_ISSs(self):
        return self.__ISSs

    def get_genes_scoring(self):
        return self.__genes_scoring

    def get_genes_function_associated(self):
        return self.__genes_function_associated
    
    def get_genes_tlod(self):
        return self.__genes_tlod
    
    def get_genes_qss(self):
        return self.__genes_qss
    
    def get_genes_big_qss(self):
        return self.__genes_big_qss
    
    def get_cytogenetic_features(self):
        return self.__cytogenetic_features

    def get_flags(self):
        return self.__flags

    def set_ages(self, value):
        if isinstance(value, pd.Series):
            self.__ages = value
        else:
            raise Exception("Ages must be a Series")
        
    def set_ageRisk(self, value):
        if isinstance(value, pd.Series):
            self.__ageRisk = value
        else:
            raise Exception("Ages must be a Series")
        
    def set_ISSs(self, value):
        if isinstance(value, pd.Series):
            self.__ISSs = value
        else:
            raise Exception("ISS must be a Series")

    def set_genes_scoring(self, value):
        if isinstance(value, pd.DataFrame):
            self.__genes_scoring = value
        else:
            raise Exception("Genes scoring must be a dataframe")

    def set_genes_function_associated(self, value):
        if isinstance(value, pd.DataFrame):
            self.__genes_function_associated = value
        else:
            raise Exception("Genes function associated must be a dataframe")
    
    def set_genes_tlod(self, value):
        if isinstance(value, pd.DataFrame):
            self.__genes_tlod = value
        else:
            raise Exception("Genes tlod must be a dataframe")
        
    def set_genes_qss(self, value):
        if isinstance(value, pd.DataFrame):
            self.__genes_qss = value
        else:
            raise Exception("Genes QSS must be a dataframe")    
    
            
    def set_genes_big_qss(self, value):
        if isinstance(value, pd.DataFrame):
            self.__genes_big_qss = value
        else:
            raise Exception("Genes QSS must be a dataframe") 
    
    def set_cytogenetic_features(self, value):
        if isinstance(value, pd.DataFrame):
            self.__cytogenetic_features = value
        else:
            raise Exception("Cytogenetic features must be a dataframe")
    
    def set_flags(self, value):
        if value is not None:
            if isinstance(value, pd.Series):
                self.__flags = value
            else:
                raise Exception("Flags must be a Series")
    
    def getFullDataframe(self, withPatients=True, withFlags=True, withCytogenetics=True):
        fulldf = [self.__patients.copy()]
        if self.__ages is not None:
            fulldf.append(self.__ages)
        if self.__ageRisk is not None:
            self.__ageRisk.name = "D_Age_Risk"
            fulldf.append(self.__ageRisk)
        if self.__ISSs is not None:
            fulldf.append(self.__ISSs)
        #if self.__genes_scoring is not None:
            #fulldf.append(self.__genes_scoring)
        if self.__genes_function_associated is not None:
            fulldf.append(self.__genes_function_associated)
        if self.__genes_tlod is not None:
            fulldf.append(self.__genes_tlod)
        if self.__genes_qss is not None:
            fulldf.append(self.__genes_qss)
        if self.__genes_big_qss is not None:
            fulldf.append(self.__genes_big_qss)
        if self.__cytogenetic_features is not None and withCytogenetics:
            fulldf.append(self.__cytogenetic_features)
        if self.__flags is not None and withFlags:
            fulldf.append(self.__flags)
        fulldf = pd.concat(fulldf, axis=1)
        if not withPatients:
            fulldf.drop("Patient", axis=1, inplace=True)
        return fulldf        