import random
import pandas as pd
import numpy as np 
import colorama
from colorama import Fore
pd.options.mode.chained_assignment = None
try:
    '''Finding Things and data research--------------->'''
    def Find_Categorical_col(df):
        ''' Input: pd.Dataframe()
            Output: pd.DataFrame() with Categroical Variables. 
            Parameters: pandas.DataFrame
            Description: This will return all categorical vairables from data frame.
            It should only be used for bigger datasets and should not be used for smaller datasets.'''
        col_names = []
        dty_object = df.select_dtypes('object')
        dty_num = df.select_dtypes('number')
        dty_num_index = list(dty_num.columns)
        dty_object_index = list(dty_object.columns)
        
        for i in range(len(dty_num_index)):
            # '''percent val gives 25% length of data if its a category the len of category must be less then overall data len'''
            dty_index = list(dty_num.columns)
            temp = list(dty_num[dty_index[i]].unique())
            percentval = len(dty_num[dty_index[i]])/4 
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        
        for i in range(len(dty_object_index)):
            # '''percent val gives 25% length of data if its a category the len of category must be less then overall data len'''
            dty_index = list(dty_object.columns)
            temp = list(dty_object[dty_index[i]].unique())
            percentval = len(dty_object[dty_index[i]])/4 
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        
        return col_names
    
    def Find_Categorical_return_df(df):
        '''  Input: pd.Dataframe()
             Output: pd.DataFrame() # with Categroical Variables.
             This will return all categorical vairables and converts into data frame.
             It should only be used for bigger datasets and should not be used for smaller datasets.'''
        col_names = []
        dty_object = df.select_dtypes('object')
        dty_num = df.select_dtypes('number')
        dty_num_index = list(dty_num.columns)
        dty_object_index = list(dty_object.columns)
        
        for i in range(len(dty_num_index)):
            dty_index = list(dty_num.columns)
            temp = list(dty_num[dty_index[i]].unique()) #We use unique to get only unique values we can also use value counts but if a data has similar counts it will be an issue.
            percentval = len(dty_num[dty_index[i]])/4 #percent val gives 25% length of data if its a category the len of category must be less then overall data len
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        for i in range(len(dty_object_index)):
            dty_index = list(dty_object.columns)
            temp = list(dty_object[dty_index[i]].unique())
            percentval = len(dty_object[dty_index[i]])/4 #percent val gives 25% length of data if its a category the len of category must be less then overall data len
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        df2 = df[col_names]
        return df2
    
    def Find_Categorical_dtype_num(df):
        ''' Input: pd.Dataframe()
            Output: pd.DataFrame() # with Categroical Variables.
            Description: This will return numerical dtype categorical vairables from data frame.
            It should only be used for bigger datasets and should not be used for smaller datasets.''' 
        col_names = []
        dty_num = df.select_dtypes('number')
        dty_num_index = list(dty_num.columns)
        
        for i in range(len(dty_num_index)):
            dty_index = list(dty_num.columns)
            temp = list(dty_num[dty_index[i]].unique())
            percentval = len(dty_num[dty_index[i]])/4 #percent val gives 25% length of data if its a category the len of category must be less then overall data len
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        
        return col_names
    
    def Find_Categorical_dtype_obj(df, force = 'no'):
        '''
        Input: df = pd.DataFrame(), force = String 
        Output: pd.DataFrame()
        Params: force: yes or no
        Description: if force = yes it will only give you object values. if no then it will return only categorical data not all object data for example if you have a id as an object type id is unique for each row hence its not a categorical data type or have any category it will not include those data set if force = no.
        This will return text or object categorical vairables from data frame.
        It should only be used for bigger datasets and should not be used for smaller datasets.'''
        col_names = []
        dty_object = df.select_dtypes('object')
        dty_object_index = list(dty_object.columns)
        if(force == 'yes'):
            col_names_new = df.select_dtypes('object')
            return col_names_new.columns
        for i in range(len(dty_object_index)):
            dty_index = list(dty_object.columns)
            temp = list(dty_object[dty_index[i]].unique())
            percentval = len(dty_object[dty_index[i]])/4 #percent val gives 25% length of data if its a category the len of category must be less then overall data len
            if(len(temp)<percentval and len(temp)!=percentval):
                col_names.append(dty_index[i])
        return col_names
    
    def Capture_NaN(dataframe,feature = ''):
        '''
        Input: pd.DataFrame()
        Output: pd.DataFrame()
        Params: dataframe = ,feature = 
        Prerequisite: dataframe parameter cannot stay empty.
        Description: Capture_NaN will perform Capturing of null values and will store it in a new feature naming : ['your_feature_name' + _Capture]
        
        '''
        string = ''
        listdata = []
        #-----> it will execute if there are no features given.
        if(len(feature) == 0):
            #it will do for all feature that contains null values.
            index = np.where(dataframe.isnull().sum()>0)
            index = list(np.array(index).flatten())
            feature = list(dataframe.columns)
            for i in index:
                dataframe[feature[i]+"_Capture"] = np.where(dataframe[feature[i]].isnull(),1,0)
       
    # -----> it will execute if there are features given.
        else:
            if type(feature) == type(string):
                dataframe[feature+"_Capture"]= np.where(dataframe[feature].isnull(),1,0) #1 = NaN , 0 = Non Null Values.
            elif type(feature) == type(listdata): 
                for i in feature:
                    dataframe[i+'_Capture'] = np.where(dataframe[i].isnull(),1,0)
        return dataframe
        
        
    def Find_Missing_percentage(dataframe):
        '''
            Input: pd.DataFrame()
            Output: features: -> Missing Value Percentage
            Params: pd.DataFrame()
            Description: Find Out Only Missing Values in respect to Percentage. It will return you all the percentage value of Null columns. 
        '''
        features = dataframe.columns
        index = np.where(dataframe.isnull().sum()>0)
        index = np.array(index).flatten()
        index = [int(i) for i in index]
        for i in index: 
            print(f'{features[i]} = {dataframe[features[i]].isnull().mean()}')
    
    def Find_Missing_col(dataframe):
        '''Lets Find Out Only Missing Values in respect to Percentage.
            Input:
            Output:
            Params:
            Prerequisite:
            Description:
        '''
        features = dataframe.columns
        index = np.where(dataframe.isnull().sum()>0)
        index = np.array(index).flatten()
        index = [int(i) for i in index]
        for i in index: 
            print(f'{features[i]}')
            
    def Find_corr_drop(dataframe,features,thresh):
            ''' 
            Input: pd.DataFrame(), features, thresh
            Output: updated pd.dataframe()
            Params:pd.DataFrame(), feautres = pd.Series(), thresh = floating decimal values
            Description: This Will check the correlation of features and will delete based on threshold for particular feature only.
            '''
            result = {}
            final_result = []
            string = ''
            listdata = [] 
            newdf = {}
            if type(features) == type(string):
                result = dict(dataframe.corr()[features])
                index = result.keys()
                final_result = {}
                for i in index:
                    if(result[i]>thresh):
                        temp = i
                        final_result[i] = result[i]
                    fr = dict(pd.Series(final_result))
                    empty = set()
                    for keyss in fr:
                        if(fr[keyss]>thresh and keyss!=features):
                            empty.add(keyss)
                dataframe.drop(empty, axis = 1, inplace = True)
                return dataframe
                
            else:
                print(f'{Fore.RED}Please Check All The Parameters And Try Again.')
        
    def Find_corr_drop_all(dataframe,thresh,feature = ''):
            '''
            Input: pd.DataFrame(), features, thresh, 
            Output: updated pd.dataframe()
            Params:pd.DataFrame(), feautres = pd.Series(), thresh = floating decimal values , 
            Description: This Will check the correlation of features and will delete based on threshold for all.
            thresh = accepts percentage values in decimal
            sign = accpets +ve , -ve string values.
            dataframe = pandas dataframe pd.DataFrame()
            This will return all columns for certain defined +ve or -ve cor-relation.    '''   

            if feature!='':
                temp_df = dataframe[feature]
                corr_matrix = dataframe.corr().abs()
                # Select upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find features with correlation greater than 0.95
                to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
                # Drop features 
                dataframe.drop(to_drop, axis=1, inplace=True)
                dataframe[feature] = temp_df
                return dataframe
            elif feature=="":
                corr_matrix = dataframe.corr().abs()
                # Select upper triangle of correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find features with correlation greater than 0.95
                to_drop = [column for column in upper.columns if any(upper[column] > thresh)]
                # Drop features 
                dataframe.drop(to_drop, axis=1, inplace=True)
                return dataframe
                
            else:
                print(f'{Fore.RED}Please Check All The Parameters And Try Again.')
        
    def Find_corr(dataframe,features,thresh,sign):
            '''
            Input: pd.DataFrame(), features, thresh, sign
            Output: updated pd.dataframe()
            Params:pd.DataFrame(), feautres = pd.Series(), thresh = floating decimal values , sign = '+ve', '-ve', or all = ''
            Description: This Will check the correlation of features and will delete based on threshold.
            thresh = accepts percentage values in decimal
            sign = accpets +ve , -ve string values.
            dataframe = pandas dataframe pd.DataFrame()
            This will return all columns for certain defined +ve or -ve cor-relation.    '''       
            result = {}
            final_result = []
            string = ''
            listdata = [] 
            newdf = {}
            if(thresh == 12120):
                if(sign == ''):
                    if type(features) == type(string):
                        result = dict(dataframe.corr()[features])
                        index = result.keys()
                        final_result = {}
                        for i in index:    
                            temp = i
                            final_result[i] = result[i]
                        return pd.Series(final_result)
                    
                    if type(features) == type(listdata):
                        for feature in features:
                                result = dict(dataframe.corr()[feature])
                                index = result.keys()
                                final_result = {}
                                for i in index:
                                    temp = i
                                    final_result[i] = result[i]
                                
                                newdf[feature] = final_result
                               # print(f'{feature} :\n{pd.Series(final_result)}\n')
                        return pd.DataFrame(newdf) 
            
            if(thresh == 12120):
                thresh = 0
            #Its for +ve correlation finding --------------------->
            if(sign == '+ve'):
                if type(features) == type(string):
                    result = dict(dataframe.corr()[features])
                    index = result.keys()
                    final_result = {}
                    for i in index:
                        if(result[i]>thresh):
                            temp = i
                            final_result[i] = result[i]
                    return pd.Series(final_result)
                
                if type(features) == type(listdata):
                    for feature in features:
                            result = dict(dataframe.corr()[feature])
                            index = result.keys()
                            final_result = {}
                            for i in index:
                                if(result[i]>thresh):
                                    temp = i
                                    final_result[i] = result[i]
                            
                            newdf[feature] = final_result
#                            print(f'{feature} :\n{pd.Series(final_result)}\n')
                    return pd.DataFrame(newdf) 
            if(sign == '-ve'):
                if type(features) == type(string):
                    result = dict(dataframe.corr()[features])
                    index = result.keys()
                    final_result = {}
                    for i in index:
                        if(result[i]<thresh):
                            temp = i
                            final_result[i] = result[i]
                    return pd.Series(final_result)
                
                if type(features) == type(listdata):
                    for feature in features:
                            result = dict(dataframe.corr()[feature])
                            index = result.keys()
                            final_result = {}
                            for i in index:
                                if(result[i]<thresh):
                                    temp = i
                                    final_result[i] = result[i]
                            newdf[feature] = final_result
#                           print(f'{feature} :\n{pd.Series(final_result)}\n')
                    return pd.DataFrame(newdf) 
            
            elif(thresh != 12120):
                if(sign == ''):
                    if type(features) == type(string):
                        result = dict(dataframe.corr()[features])
                        index = result.keys()
                        final_result = {}
                        for i in index:    
                            if(result[i]<thresh):
                                temp = i
                                final_result[i] = result[i]
                        return pd.Series(final_result)
                    
                    if type(features) == type(listdata):
                        for feature in features:
                                result = dict(dataframe.corr()[feature])
                                index = result.keys()
                                final_result = {}
                                for i in index:
                                    if(result[i]<thresh):
                                        temp = i
                                        final_result[i] = result[i]
                                
                                newdf[feature] = final_result
#                                 print(f'{feature} :\n{pd.Series(final_result)}\n')
                        return pd.DataFrame(newdf) 
              
                
                
                        
        
     
    
                
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
            self.dataframe = dataframe
            self.perc = perc
            self.dtype = dtype
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
            self.dataframe = dataframe
            self.dtype = dtype
            columns = dataframe.select_dtypes(dtype)
            columns = list(columns.columns)
            for column in columns:
                if(dataframe[column].isnull().sum()>0):
                    dataframe.drop(column,axis = 1, inplace = True)
            
            return dataframe
        
       
    
    
    class Impute:
        def remove_outliers(self,dataframe,features):
            '''
            Input: pd.DataFrame(), features = pd.Series() => List, String
            Output: pd.DataFrame()
            Params:  pd.DataFrame(), features = pd.Series() that we need to remove outliers from
            Description: It will remove all outliers from the dataframe and will give you result features can be a list or string do remember you will be losing data if you do this step  .
            
            '''
            self.dataframe = dataframe
            self.features = features
            string = ''
            listdata = []
            try:
                if type(features) == type(string):
                    Q1 = dataframe[features].quantile(0.25)
                    Q3 = dataframe[features].quantile(0.75)
                    IQR = Q3-Q1
                    lower_bracket = Q1 - 1.5*IQR
                    higher_bracket = Q3 + 1.5*IQR
                    dataframe = dataframe[(dataframe[features]>lower_bracket) & (dataframe[features]<higher_bracket)]
                    return dataframe
                if type(features) == type(listdata):
                    for feature in features:
                        Q1 = dataframe[feature].quantile(0.25)
                        Q3 = dataframe[feature].quantile(0.75)
                        IQR = Q3-Q1
                        lower_bracket = Q1 - 1.5*IQR
                        higher_bracket = Q3 + 1.5*IQR
                        dataframe = dataframe[(dataframe[feature]>lower_bracket) & (dataframe[feature]<higher_bracket)]
                    return dataframe
            except Exception as e:
                print(Fore.RED + e)

        def replace_outliers(self,dataframe,features, kind = 'median', custom = ''):
            '''
            Input: pd.DataFrame(), features = pd.Series() => List, String, kind = ['median' (default), mode, mean] , custom = 'Your custom Values'
            Output: pd.DataFrame()
            Params:  pd.DataFrame(), features = pd.Series() that we need to replace outliers from,  kind = ['median' (default), mode, mean] => you can change method using kind , custom = 'Your custom Values'
            Description: This will replace outliers from specified kind values this is best if you don't want to lose data but will impact the relationship of your data also you can specifiy your own custom values for the outliers.
            '''
            self.dataframe = dataframe
            self.features = features
            self.custom = custom
            self.kind = kind
            string = ''
            listdata = []
            try:
                if type(features) == type(string):
                    median = dataframe[features].median()
                    mode = dataframe[features].mode()
                    mean = dataframe[features].mean()
                    #Lets find out outliers----->
                    Q1 = dataframe[features].quantile(0.25)
                    Q3 = dataframe[features].quantile(0.75)
                    IQR = Q3-Q1
                    lower_bracket = Q1 - 1.5*IQR
                    higher_bracket = Q3 + 1.5*IQR
                    index = np.where(dataframe[features]>higher_bracket) + np.where(dataframe[features]<lower_bracket)
                    index = list(index[0])
                    index = [int(i) for i in index]
                    if(kind == 'median'):
                        for j in index:
                            dataframe[features][j] = median
                    if(kind == 'mode'):
                        for j in index:
                            dataframe[features][j] = mode
                    if(kind == 'mean'):
                        for j in index:
                            dataframe[features][j] = mean
                    if(kind == 'custom'):
                        if(custom == ''):
                            print(Fore.RED + 'ERROR : Enter Custom Value And Try Again')
                        if(custom != ''):
                            for j in index:
                                dataframe[features][j] = custom
                    return dataframe

                   
                if type(features) == type(listdata):
                    for feature in features:
                        median = dataframe[feature].median()
                        mode = dataframe[feature].mode()
                        mean = dataframe[feature].mean
                        # '''Lets find out outliers----->
                        Q1 = dataframe[feature].quantile(0.25)
                        Q3 = dataframe[feature].quantile(0.75)
                        IQR = Q3-Q1
                        lower_bracket = Q1 - 1.5*IQR
                        higher_bracket = Q3 + 1.5*IQR
                        index = np.where(dataframe[feature]>higher_bracket) + np.where(dataframe[feature]<lower_bracket)
                        index = list(index[0])
                        index = [int(i) for i in index]
                        if(kind == 'median'):
                            for j in index:
                                dataframe[feature][j] = median
                        if(kind == 'mode'):
                            for j in index:
                                 dataframe[feature][j] = mode
                        if(kind == 'mean'):
                            for j in index:
                                dataframe[feature][j] = mean
                        if(kind == 'custom'):
                            if(custom == ''):
                                print(Fore.RED + 'ERROR : Enter Custom Value And Try Again')
                            if(custom != ''):
                                for j in index:
                                    dataframe[feature][j] = custom
                
                    return dataframe
           
            except Exception as e:
                print(Fore.RED + e)
           
    
        def frequent_category(self,dataframe,feature):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => List, String
            Output: pd.DataFrame()
            Description: This will impute all features null values with frequent category technique or frequency counts.
            '''
            self.dataframe = dataframe
            self.feature = feature
            string = ''
            listdata = [] 
            if type(feature) == type(string):
                freq_cat = dataframe[feature].value_counts().index[0]
                dataframe[feature].fillna(freq_cat, inplace = True)
        
            if type(feature) == type(listdata):
                for features in feature:
                    freq_cat = dataframe[features].value_counts().index[0]
                    dataframe[features].fillna(freq_cat, inplace = True)
                    
            return dataframe
            
        def endofdist(self,dataframe,feature,newcol = False):
            '''
            Input: pd.DataFrame(), feature = pd.Series() =>list, String, newcol = bool
            Output: pd.DataFrame()
            Description:If there is a suspicion that the missing value is not at random then capturing that information is important.
            In this scenario, one would want to replace missing data with values that are at the tail of the distribution of the variable.
            This will impute all features null values with end of distribution technique with newcol = True you can create new column with replaced null values.
            '''
            self.dataframe = dataframe
            self.feature = feature
            self.newcol = newcol
            string = ''
            listdata = [] 
            if(str(dataframe[feature].dtype) == 'object'):
                print(Fore.RED + "Sorry Dataype Not supported")
                
            if newcol == False:
                if type(feature) == type(string):
                    extreme = dataframe[feature].mean() + 3*dataframe[feature].std()
                    dataframe[feature] = dataframe[feature].fillna(extreme)
                if type(feature) == type(listdata):
                    for i in range(len(feature)):
                        extreme = dataframe[feature[i]].mean() + 3*dataframe[feature[i]].std()
                        dataframe[feature[i]] = dataframe[feature[i]].fillna(extreme)
                    
            elif newcol == True :
                if type(feature) == type(string):
                    extreme = dataframe[feature].mean() + 3*dataframe[feature].std()
                    dataframe[feature + '_NEWCOL'] = dataframe[feature].fillna(extreme)
                elif type(feature) == type(listdata):
                    for i in range(len(feature)):
                        extreme = dataframe[feature[i]].mean() + 3*dataframe[feature[i]].std()
                        dataframe[feature[i]+'_NEWCOL'] = dataframe[feature[i]].fillna(extreme)
           
            return dataframe

       
        def rand_sample(self,dataframe,feature = ""):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => list, string
            Output: pd.DataFrame()
            Description: This will convert null values in a feature columns specified by using list or string into randomly selected values only applicable for numerical features.
            '''
            self.feature = feature
            self.dataframe = dataframe
            columns = dataframe.columns.tolist()
            index = np.where(dataframe.isnull().sum()>0)
            index = np.array(index).flatten() #As it is giving a 2d array flatten will convert it to 1d array.
            index = index.tolist()
            # '''Using List Comprehension it will be converted into integervalues.
            index = [int(i) for i in index]
            
            string = ''
            listdata = [] 
            
            if type(feature) == type(string) and feature != "":
                random_sample = dataframe[feature].dropna().sample(dataframe[feature].isnull().sum(), random_state = 0)
                random_sample.index = dataframe[dataframe[feature].isnull()].index
                dataframe.loc[dataframe[feature].isnull(),feature] = random_sample
            if type(feature) == type(listdata) and feature != "":
                for features in feature:
                    random_sample = dataframe[features].dropna().sample(dataframe[features].isnull().sum(), random_state = 0)
                    random_sample.index = dataframe[dataframe[features].isnull()].index
                    dataframe.loc[dataframe[features].isnull(),features] = random_sample
            elif(feature == ""):
                for i in index:
                    if(dataframe[columns[i]].dtype != 'O'):
                        random_sample = dataframe[columns[i]].dropna().sample(dataframe[columns[i]].isnull().sum(), random_state = 0)
                        random_sample.index = dataframe[dataframe[columns[i]].isnull()].index
                        dataframe.loc[dataframe[columns[i]].isnull(),columns[i]] = random_sample
                    else:
                        print(f"{Fore.RED} There is issue with the feature {columns[i]} datatype or feature data please check if its categorical please use rand_sample_cat")
            
            elif(feature == "" or dataframe.empty):
                print('Error Occured Please Ensure You Have Entered Everything Properly')

            return dataframe
         
        def rand_sample_cat(self,dataframe,feature):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => list, string
            Output: pd.DataFrame()
            Description: This will convert null values in a feature columns specified by using list or string into randomly selected values only applicable for categorical features.
            '''
            self.feature = feature
            self.dataframe = dataframe
            string = ''
            listdata = []
            
            if type(feature) == type(string):
                randomize = pd.Series(dataframe[feature].unique())
                randomize = randomize.dropna()
                randomize = list(randomize)
                dataframe[feature].fillna(random.choice(randomize), inplace = True)
            if type(feature) == type(listdata):
                for features in feature:
                    randomize = pd.Series(dataframe[features].unique())
                    randomize = randomize.dropna()
                    randomize = list(randomize)
                    dataframe[features].fillna(random.choice(randomize), inplace = True)
            return dataframe
                
        def nan_mean(self,dataframe,feature):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => list, string
            Output: pd.DataFrame()
            Description: Convert nan values to feature mean values.
            '''
            ''''''
            string = ''
            listdata = [] 
            self.dataframe = dataframe
            self.feature = list(feature)
            # '''It will replace columns with mean.
            if type(feature) == type(string):
                if(dataframe[feature].dtype == 'O'):
                        display(Fore.RED + "Error Occured Looks Like Feature DataType Is Wrong")
                else:
                        mean = dataframe[feature].mean()
                        dataframe[feature] = dataframe[feature].fillna(mean)
            if type(feature) == type(listdata):
                for i in range(len(feature)):
                    if(dataframe[feature[i]].dtype == 'O'):
                        display(Fore.RED + "Error Occured Looks Like Feature DataType Is Wrong")
                    else:
                        mean = dataframe[feature[i]].mean()
                        dataframe[feature[i]] = dataframe[feature[i]].fillna(mean)
                        
            return dataframe
        
        # '''single null value replace
        def nan_median(self,dataframe,feature):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => list, string
            Output: pd.DataFrame()
            Description: Convert all nan values to median of feature values.
            '''
            feature  = list(feature)
            self.dataframe = dataframe
            self.feature = list(feature)
            for i in range(len(feature)):
                median = dataframe[feature[i]].median()
                dataframe[feature] = dataframe[feature[i]].fillna(median)
            return dataframe
            
        def nan_mode(self,dataframe,feature):
            '''
            Input: pd.DataFrame(), feature = pd.Series() => list, string
            Output: pd.DataFrame()
            Description: Convert all nan values to mode of feature values
            '''
            feature = feature
            self.dataframe = dataframe
            self.feature = list(feature)
            for i in range(len(feature)):
                mode = dataframe[feature[i]].mode().tolist()[0]
                dataframe[feature[i]] = dataframe[feature[i]].fillna(mode)
            return dataframe   
    
        def nan_mean_all(self,dataframe,force = 'no'):
            '''
            Input: pd.DataFrame(), force = 'no','yes'
            Output: pd.DataFrame()
            Params: force => if specified yes then will replace mean values of categorical also which is median of categorical based on the frequency.
            Description: Will replace all nan values in datafarme with mean and force if specified yes then will replace mean values of categorical also which is median of categorical based on the frequency.
            '''
            self.dataframe = dataframe
            self.force = force
            '''It will replace all nan value columns with mean.'''
            columns = dataframe.columns.tolist()
            index = np.where(dataframe.isnull().sum()>0)
            index = np.array(index).flatten() #As it is giving a 2d array flatten will convert it to 1d array.
            index = index.tolist()
            # '''Using List Comprehension it will be converted into integervalues.
            index = [int(i) for i in index]
            for i in index:
                    if(dataframe[columns[i]].dtype != 'O'):
                        dataframe[columns[i]] = dataframe[columns[i]].fillna(dataframe[columns[i]].mean()) 
                    elif(force == 'yes'):
                        for i in index:
                            colname = dataframe.columns[i]
                            value =  pd.DataFrame(dataframe[columns[i]].str[0].value_counts()) #it will calculate frequency of all unique categories and then replace it with median.
                            dataframe[columns[i]] = dataframe[columns[i]].fillna(value[colname].median())
            return dataframe             
        
        def nan_median_all(self,dataframe):
            '''
            Input: pd.DataFrame()
            Output: pd.DataFrame()
            Description: Will replace all nan values in datafarme with median .
            '''
            self.dataframe = dataframe
            '''It will replace all nan value columns with median.'''
            columns = dataframe.columns.tolist()
            index = np.where(dataframe.isnull().sum()>0)
            index = np.array(index).flatten() #As it is giving a 2d array flatten will convert it to 1d array.
            index = index.tolist()
            #Using List Comprehension it will be converted into integervalues.
            index = [int(i) for i in index]
            for i in index:
                    median = dataframe[columns[i]].median().tolist()[0]
                    dataframe[columns[i]] = dataframe[columns[i]].fillna(median)
            return dataframe             
        
        def nan_mode_all(self,dataframe):
            '''
            Input: pd.DataFrame()
            Output: pd.DataFrame()
            Description: Will replace all nan values in datafarme with mode .
            '''
            self.dataframe = dataframe
            columns = dataframe.columns.tolist()
            lists = []
            index = np.where(dataframe.isnull().sum()>0)
            index = np.array(index).flatten()#As it is giving a 2d array flatten will convert it to 1d array.
            index = index.tolist()
            #Using List Comprehension it will be converted into integervalues.
            index = [int(i) for i in index]
            for i in index:
                mode = dataframe[columns[i]].mode().tolist()[0] #if for some reason there comes two different values in mode
                dataframe[columns[i]] = dataframe[columns[i]].fillna(mode)        
            return dataframe
        
    class Encoder:
        """
        Class For Encoders More encoding techniques will be there in the next update this is updated in version 0.0.4;
        """
        
        def mean_target(self,dataframe,feature,target, on_new_col = 'no'):
            '''
            Input: pd.DataFrame(),feature,target,on_new_col
            Output: pd.DataFrame(),_groups
            Description: Will encode values of categorical feature in datafarme with mean target ordinal encding technique .
            
            '''
            _groups = pd.DataFrame() 
            self.dataframe = dataframe
            self.feature = feature
            self.target = target
            self.on_new_col = on_new_col
            self._groups = _groups
            string = ''
            error = 'Please Check your parameters again or check documentation if you are not clear about the use.'
            
            if on_new_col == 'yes' and type(feature) == type(string):
                mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
                _groups[feature] = dataframe[feature]
                dataframe['Mean_Ordinal_Encode'] = dataframe[feature].map(mean_ordinal)
                _groups['Encoded Values']  = dataframe['Mean_Ordinal_Encode']
                dataframe[feature] = dataframe['Mean_Ordinal_Encode']
                
                return dataframe
            elif on_new_col == 'no' and type(feature) == type(string):
                _groups[feature] = dataframe[feature]
                mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
                _groups[feature] = df[feature].map(mean_ordinal)
                dataframe[feature] = _groups[feature]
                return dataframe
            else:
                return error
        
        
except Exception as e:
       print(e)