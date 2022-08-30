import pandas as pd
import numpy as np 
from colorama import Fore

class Dropnan:
    '''Drop Null Columns Using Properties.
    Initalized as 
    import datawrang as dw
    dropnan = dw.Dropnan()
    dropnan.percentage(df,0.3)
    '''
    def percentage(self,dataframe,perc):  
        '''
        Input: pd.DataFrame(), perc = float
        Output: pd.DataFrame()
        Params: pd.DataFrame(), perc = percentage => float
        Description:
        This will remove all null values columns for certain defined percentage.   
        percentage are given in decimal format for eg 0.2 => 20%.'''
        self.dataframe = dataframe
        self.perc_value = perc
        
        if perc == '' or dataframe.empty:
            print(Fore.RED + 'Error Occured Please Check If passed empty datasets ')
       
        percindex = np.where(dataframe.isnull().mean()>perc) # collecting index values for all the null columns that have percentage of null values greater then user defined.
        percindex = np.array(percindex).flatten()
        percindex = [int(i) for i in percindex]
        columns = dataframe.columns
        for i in percindex:
            #To remove percentage we will iterate through collected index.
             dataframe.drop(columns[i],axis = 1,inplace = True)
       
        
        return dataframe
   
    def auto_perc(self,dataframe):    
        ''' 
        Input: pd.DataFrame()
        Output: pd.DataFrame()
        Params: pd.DataFrame()
        Description:
        This will remove all null value columns from dataframe which have more then 30 percent null values. '''            
        self.dataframe = dataframe
        perc = 0.3
        percindex = np.where(dataframe.isnull().mean()>perc) # collecting index values for all the null columns that have percentage of null values greater then user defined.
        percindex = np.array(percindex).flatten()
        percindex = [int(i) for i in percindex]
        columns = dataframe.columns
        for i in percindex:
            #To remove percentage we will iterate through collected index.
             dataframe.drop(columns[i],axis = 1,inplace = True)
       
    #percentage in decimal if >35 will remove those columns
        return dataframe

    def dtype_perc(self,dataframe,perc,dtype):    
        ''' 
        Input: pd.DataFrame(), perc = float, dtype = string => float,int..
        Output: pd.DataFrame()
        Params: pd.DataFrame(), perc = percentage => float, dtype = 'float', 'object', 'int'
        Description:
        This will remove all null values columns for certain defined percentage and dtype.   
        percentage are given in decimal format for eg 0.2 => 20% defined dtype should be an actual data type value. '''          
        columns = dataframe.select_dtypes(dtype)
        columns = columns.index
        percindex = np.where(dataframe.loc[columns].isnull().mean()>perc) #collecting index values for all the null columns that have percentage of null values greater then user defined.
        percindex = np.array(percindex).flatten()
        percindex = [int(i) for i in percindex]
        columns = dataframe.columns   
        for i in percindex:
            #To remove percentage we will iterate through collected index.
              dataframe.drop(columns[i],axis = 1,inplace = True)
       
    #percentage in decimal if >35 will remove those columns
        return dataframe

   
    def only_dtype(self,dataframe,dtype):    
        ''' 
        Input: pd.DataFrame(), dtype = float
        Output: pd.DataFrame()
        Params: pd.DataFrame(), dtype = DataType  => float,int,object
        Description:
        This will remove all null values columns for certain defined datatypes only.'''            
        columns = dataframe.select_dtypes(dtype)
        columns = list(columns.columns)
        for column in columns:
            if(dataframe[column].isnull().sum()>0):
                dataframe.drop(column,axis = 1, inplace = True)
        
        return dataframe
    
   
