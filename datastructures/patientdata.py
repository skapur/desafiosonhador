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
        self.__ISSs = None
        self.__genes_scoring = None
        self.__genes_function_associated = None
        self.__flags = None

    def get_dataset_origin(self):
        return self.get_dataset_origin()

    def get_ages(self):
        return self.__ages
    
    def get_ISSs(self):
        return self.__ISSs

    def get_genes_scoring(self):
        return self.__genes_scoring

    def get_genes_function_associated(self):
        return self.__genes_function_associated

    def get_flags(self):
        return self.__flags

    def set_ages(self, value):
        if isinstance(value, pd.Series):
            self.__ages = value
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
    
    def set_flags(self, value):
        if isinstance(value, pd.Series):
            self.__flags = value
        else:
            raise Exception("Flags must be a Series")
    
    def getFullDataframe(self):
        fulldf = [self.__patients.copy()]
        if self.__ages is not None:
            fulldf.append(self.__ages)
        if self.__ISSs is not None:
            fulldf.append(self.__ISSs)
        if self.__genes_scoring is not None:
            fulldf.append(self.__genes_scoring)
        if self.__genes_function_associated is not None:
            fulldf.append(self.__genes_function_associated)
        if self.__flags is not None:
            fulldf.append(self.__flags)
        fulldf = pd.concat(fulldf, axis=1)
        return fulldf        