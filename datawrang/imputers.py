import pandas as pd 
import numpy as np 
import random
from colorama import Fore

class Impute:
    def remove_outliers(self,dataframe,features):
        '''
        Input: pd.DataFrame(), features = pd.Series() => List, String
        Output: pd.DataFrame()
        Params:  pd.DataFrame(), features = pd.Series() that we need to remove outliers from
        Description: It will remove all outliers from the dataframe and will give you result features can be a list or string do remember you will be losing data if you do this step  .
        
        '''
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
        string = ''
        listdata = []
        a = np.array([5,5])
        a = pd.DataFrame(a)
        a = a[0].astype('category')
        if(type(features) == str):
            if(dataframe[features].dtype == 'O' or dataframe[features].dtype.str == a.dtype.str):
                    raise TypeError('Error I1x001: Please check your given features datatype features should not be categorical only numbers, float,int dataypes are allowed for replacement')
        elif(type(features)== list):
            for feature in features:
                if(dataframe[feature].dtype == 'O' or dataframe[feature].dtype.str == a.dtype.str):
                    raise TypeError('Error I1x001: Please check your given features datatype features should not be categorical only numbers, float,int dataypes are allowed for replacement')
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

        # '''It will replace columns with mean.
        if type(feature) == type(string):
            if(dataframe[feature].dtype == 'O'):
                    print(Fore.RED + "Error Occured Looks Like Feature DataType Is Wrong")
            else:
                    mean = dataframe[feature].mean()
                    dataframe[feature] = dataframe[feature].fillna(mean)
        if type(feature) == type(listdata):
            for i in range(len(feature)):
                if(dataframe[feature[i]].dtype == 'O'):
                    print(Fore.RED + "Error Occured Looks Like Feature DataType Is Wrong")
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
        if type(feature) == str:
            median = dataframe[feature].median()
            dataframe[feature] = dataframe[feature].fillna(median)
        if(type(feature)==list):
            for i in range(len(feature)):
                median = dataframe[feature[i]].median()
                dataframe[feature[i]].fillna(median,inplace = True)
        return dataframe
        
    def nan_mode(self,dataframe,feature):
        '''
        Input: pd.DataFrame(), feature = pd.Series() => list, string
        Output: pd.DataFrame()
        Description: Convert all nan values to mode of feature values
        '''
        if type(feature) == str:
            mode = dataframe[feature].mode().tolist()[0]
            dataframe[feature] = dataframe[feature].fillna(mode)
        if type(feature) == list:
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
        
        '''It will replace all nan value columns with mean.'''
        a = np.array([5,5])
        a = pd.DataFrame(a)
        a = a[0].astype('category')
        
        columns = dataframe.columns.tolist()
        index = np.where(dataframe.isnull().sum()>0)
        index = np.array(index).flatten() #As it is giving a 2d array flatten will convert it to 1d array.
        index = index.tolist()
        # '''Using List Comprehension it will be converted into integervalues.
        index = [int(i) for i in index]
        if(force == 'no'):
            for i in index:
                if(dataframe[columns[i]].dtype != 'O' and dataframe[columns[i]].dtype.str != a.dtype.str):
                    dataframe[columns[i]] = dataframe[columns[i]].fillna(dataframe[columns[i]].mean()) 
        elif(force == 'yes'):
            for i in index:
                if(dataframe[columns[i]].dtype != 'O' and dataframe[columns[i]].dtype.str != a.dtype.str):
                    dataframe[columns[i]] = dataframe[columns[i]].fillna(dataframe[columns[i]].mean())
                else:
                    colname = dataframe.columns[i]
                    value =  pd.DataFrame(dataframe[columns[i]].value_counts()) #it will calculate frequency of all unique categories and then replace it with median.
                    dataframe[columns[i]].fillna(value[colname].median(), inplace = True)
        return dataframe             
    
    def nan_median_all(self,dataframe):
        '''
        Input: pd.DataFrame()
        Output: pd.DataFrame()
        Description: Will replace all nan values in datafarme with median .
        '''
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
    
