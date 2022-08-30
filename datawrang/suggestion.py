import pandas as pd
import numpy as np 
pd.options.mode.chained_assignment = None

class Suggest:
    def __init__(self,dataframe,feature):
        self.dataframe = dataframe
        self.feature = feature
    
    def mar_mnar(self):
        pass