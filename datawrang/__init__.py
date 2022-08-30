import pandas as pd
import numpy as np 
from colorama import Fore
from encoders import Encoder
from pipelines import Pipeline
from drop import Dropnan
from suggestion import Suggest
pd.options.mode.chained_assignment = None
integer = 0
'''Finding Things and data research--------------->'''
def Find_Categorical_col(dataframe):
    ''' Input: pd.Dataframe()
        Output: pd.DataFrame() with Categroical Variables. 
        Parameters: pandas.DataFrame
        Description: This will return all categorical vairables from data frame.
        It should only be used for bigger datasets and should not be used for smaller datasets.'''
    col_names = []
    dty_object = dataframe.select_dtypes('object')
    dty_num = dataframe.select_dtypes('number')
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

def Find_fpr_tpr_prec_spec(y_true, y_prob, thresholds = [0.5]):
    '''
        Input: y_true, y_prob, thresholds
        Output: False Postive Rate , True Positive Rate(Recall), Precision, Specificity
        Params:y_true = actual values , y_prob =  probalbity (predict_proba), thresholds (By default = 0.5) you can give a list of values or single value, 
        Description: This will return all fpr tpr precision and specifity values for different thresholds in a list. to capture it use
        ```
        >>> fpr, tpr, precision, specificity = datawrang.Find_fpr_tpr_prec_spec(y_true,y_prob)
        >>> # If you just want to get precison and recall use fpr_precision, or if you want fpr and tpr only use fpr_tpr
        ```
        '''   
    try:
        fpr = []
        tpr = []
        precision = []
        specificity = []
        for thresh in thresholds:
            y_pred = np.where(y_prob >= thresh, 1, 0)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))
            precision.append(tp / (tp + fp))
            specificity.append(tn / (tn + fp))
        return fpr, tpr, precision, specificity
    except Exception as e:
        print(f"Error occured check parameter values or if you are using SVC Model make sure probalibty is True {e}")


def Find_fpr_tpr(y_true, y_prob, thresholds = [0.5]):
    '''
        Input: y_true, y_prob, thresholds
        Output: False Postive Rate , True Positive Rate(Recall)
        Params:y_true = actual values , y_prob =  probalbity (predict_proba), thresholds (By default = 0.5) you can give a list of values or single value, 
        Description: This will return all fpr tpr precision and specifity values for different thresholds in a list. to capture it use
        ```
        >>> fpr, tpr, precision, specificity = datawrang.Find_fpr_tpr_prec_spec(y_true,y_prob)
        >>> # If you just want to get precison and recall use tpr_prec, or if you want fpr and tpr only use fpr_tpr, for all use Find_fpr_tpr_prec_spec
        ```
        '''   
    try:
        fpr = []
        tpr = []
        for thresh in thresholds:
            y_pred = np.where(y_prob >= thresh, 1, 0)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))
        return fpr, tpr
    except Exception as e:
        print(f"Error occured check parameter values or if you are using SVC Model make sure probalibty is True {e}")


def Find_tpr_prec(y_true, y_prob, thresholds = [0.5]):
    '''
        Input: y_true, y_prob, thresholds
        Output: True Positive Rate(Recall), Precision
        Params:y_true = actual values , y_prob =  probalbity (predict_proba), thresholds (By default = 0.5) you can give a list of values or single value, 
        Description: This will return all fpr tpr precision and specifity values for different thresholds in a list. to capture it use
        ```
        >>> fpr, tpr, precision, specificity = datawrang.Find_fpr_tpr_prec_spec(y_true,y_prob)
        >>> # If you just want to get precison and recall use fpr_precision, or if you want fpr and tpr only use fpr_tpr, for all use Find_fpr_tpr_prec_spec
        ```
        '''   
    try:
        precision = []
        tpr = []
        for thresh in thresholds:
            y_pred = np.where(y_prob >= thresh, 1, 0)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            
            precision.append(tp / (tp + fp))
            tpr.append(tp / (tp + fn))
        return tpr,precision
    except Exception as e:
        print(f"Error occured check parameter values or if you are using SVC Model make sure probalibty is True {e}")



        
        
    
def Find_Categorical_return_df(dataframe):
    '''  Input: pd.Dataframe()
         Output: pd.DataFrame() # with Categroical Variables.
         This will return all categorical vairables and converts into data frame.
         It should only be used for bigger datasets and should not be used for smaller datasets.'''
    col_names = []
    dty_object = dataframe.select_dtypes('object')
    dty_num = dataframe.select_dtypes('number')
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
    df2 = dataframe[col_names]
    return df2

def Find_Categorical_dtype_num(dataframe):
    ''' Input: pd.Dataframe()
        Output: pd.DataFrame() # with Categroical Variables.
        Description: This will return numerical dtype categorical vairables from data frame.
        It should only be used for bigger datasets and should not be used for smaller datasets.''' 
    col_names = []
    dty_num = dataframe.select_dtypes('number')
    dty_num_index = list(dty_num.columns)
    
    for i in range(len(dty_num_index)):
        dty_index = list(dty_num.columns)
        temp = list(dty_num[dty_index[i]].unique())
        percentval = len(dty_num[dty_index[i]])/4 #percent val gives 25% length of data if its a category the len of category must be less then overall data len
        if(len(temp)<percentval and len(temp)!=percentval):
            col_names.append(dty_index[i])
    
    return col_names

def Find_Categorical_dtype_obj(dataframe, force = 'no'):
    '''
    Input: dataframe = pd.DataFrame(), force = String 
    Output: pd.DataFrame()
    Params: force: yes or no
    Description: if force = yes it will only give you object values. if no then it will return only categorical data not all object data for example if you have a id as an object type id is unique for each row hence its not a categorical data type or have any category it will not include those data set if force = no.
    This will return text or object categorical vairables from data frame.
    It should only be used for bigger datasets and should not be used for smaller datasets.'''
    col_names = []
    dty_object = dataframe.select_dtypes('object')
    dty_object_index = list(dty_object.columns)
    if(force == 'yes'):
        col_names_new = dataframe.select_dtypes('object')
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
#                           print(f'{feature} :\n{pd.Series(final_result)}\n')
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
#                          print(f'{feature} :\n{pd.Series(final_result)}\n')
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
#                                print(f'{feature} :\n{pd.Series(final_result)}\n')
                    return pd.DataFrame(newdf) 
          

        
    
