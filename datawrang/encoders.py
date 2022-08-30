import random
import pandas as pd
import numpy as np 
import warnings
from colorama import Fore
pd.options.mode.chained_assignment = None
integer = 0

class Encoder:
    """
    Class For Encoders More encoding techniques will be there in the next update this is updated in version 0.0.4;
    """
    def __init__(self,dataframe = pd.DataFrame()):
        self.dataframe = dataframe
        self._groups = {}
        
    def mean_target(self,feature,target, on_new_col = 'no'):
        '''
        Input: pd.DataFrame(),feature,target,on_new_col
        Output: pd.DataFrame(),_groups(Values associated by encoding.)
        Description: Will encode values of categorical feature in datafarme with mean target ordinal encding technique .
        
        '''
        if self.dataframe.empty:
               raise TypeError("enter a dataframe in encoder") 
        _groups = {}
        dataframe = self.dataframe
        string = ''
        error = 'Please Check your parameters again or check documentation if you are not clear about the use.'
        if(type(feature) == str):
            cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
            cat_check = len(dataframe[feature].value_counts())/cat_check
            cat_check = cat_check*100
            if cat_check > 20:
                warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
        
        if on_new_col == 'yes' and type(feature) == type(string) or type(feature)==type(integer):
            mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
            _groups[feature] = dataframe[feature]
            dataframe[f'{feature}_Mean_Ordinal_Encode'] = dataframe[feature].map(mean_ordinal)
            _groups['Encoded Values']  = dataframe[f'{feature}_Mean_Ordinal_Encode']
            self._groups[feature] = mean_ordinal
            self.datafarme = dataframe
            return dataframe
        
        elif on_new_col == 'no' and type(feature) == type(string) or type(feature)==type(integer):
            _groups[feature] = dataframe[feature]
            mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
            _groups[feature] = dataframe[feature].map(mean_ordinal)
            dataframe[feature] = _groups[feature]
            self._groups[feature] = mean_ordinal
            self.datafarme = dataframe
            return dataframe
        
        # For list passed if any.
        
        elif on_new_col == 'no' and type(feature) == list:
            for feature in feature:
                cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
        
                _groups[feature] = dataframe[feature]
                mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
                _groups[feature] = dataframe[feature].map(mean_ordinal)
                dataframe[feature] = _groups[feature]
                self._groups[feature] = mean_ordinal
                self.datafarme = dataframe
            return dataframe
        
        elif on_new_col == 'yes' and type(feature) == list:
            for feature in feature:
                cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
                mean_ordinal = dataframe.groupby([feature])[target].mean().to_dict()
                _groups[feature] = dataframe[feature]
                dataframe[f'{feature}_Mean_Ordinal_Encode'] = dataframe[feature].map(mean_ordinal)
                _groups['Encoded Values']  = dataframe[f'{feature}_Mean_Ordinal_Encode']
                self._groups[feature] = mean_ordinal
                self.datafarme = dataframe
            return dataframe
        
        else:
            return error
        
    def label_encoder(self,feature,on_new_col = 'no'):
        '''
        Input: pd.DataFrame(),feature,on_new_col
        Output: pd.DataFrame(),_groups(Values associated by encoding.)
        Description: Will encode values of feature in datafarme with labels associated to each unique element in feature .
        '''
        if self.dataframe.empty:
               raise TypeError("Error EX2001 encoder dataframe is empty enter a dataframe in encoder") 
        dataframe = self.dataframe
        string = ''
        error = 'Please Check your parameters again or check documentation if you are not clear about the use.'
        
        #Processing
        if(on_new_col == 'no' and type(feature) == type(string) or type(feature)==type(integer)):
            labels = dataframe[feature].unique()
            for i,label in enumerate(labels):
                dataframe[feature].replace(label,i,inplace = True,regex=True)
                
                if(self._groups.get(feature,False) == False):
                    self._groups[feature] = {label:i}
                else:
                    self._groups[feature].update({label:i})      
                
        elif(on_new_col == 'yes' and type(feature) == type(string) or type(feature)==type(integer)):
            labels = dataframe[feature].unique()
            dataframe[f'{feature}_Label_encode'] = dataframe[feature]
            for i,label in enumerate(labels): 
                dataframe[f'{feature}_Label_encode'].replace(label,i,inplace = True,regex = True)
                if(self._groups.get(feature,False) == False):
                    self._groups[feature] = {label:i}
                else:
                    self._groups[feature].update({label:i})
                    
        elif(on_new_col == 'yes' and type(feature) == list):
            for feature in feature:
                labels = dataframe[feature].unique()
                dataframe[f'{feature}_Label_encode'] = dataframe[feature]
                for i,label in enumerate(labels): 
                    dataframe[f'{feature}_Label_encode'].replace(label,i,inplace = True,regex = True)
                    if(self._groups.get(feature,False) == False):
                        self._groups[feature] = {label:i}
                    else:
                        self._groups[feature].update({label:i})
        
        elif(on_new_col == 'yes' and type(feature) == list):
            for feature in feature:
                labels = dataframe[feature].unique()
                dataframe[f'{feature}_Label_encode'] = dataframe[feature]
                for i,label in enumerate(labels): 
                    dataframe[f'{feature}_Label_encode'].replace(label,i,inplace = True,regex = True)
                    if(self._groups.get(feature,False) == False):
                        self._groups[feature] = {label:i}
                    else:
                        self._groups[feature].update({label:i})                    

        else:
            return error

        self.datafarme = dataframe
        return dataframe
        
    def frequency_encoder(self,feature,on_new_col = 'no'):
        '''
        Input: pd.DataFrame(),feature,on_new_col
        Output: pd.DataFrame(),_groups(Values associated by encoding.)
        Description: Will encode values of feature in datafarme with freqeuncy encoding techniques; frequency associated to each unique element in feature .
        '''
        if self.dataframe.empty:
            raise TypeError("enter a dataframe in encoder") 
        dataframe = self.dataframe
        string = ''
        error = 'Please Check your parameters again or check documentation if you are not clear about the use.'
        if type(feature) == str:
            cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
            cat_check = len(dataframe[feature].value_counts())/cat_check
            cat_check = cat_check*100
            if cat_check > 20:
                warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")

        if(on_new_col == 'no' and type(feature) == type(string) or type(feature)==type(integer)):
            value_counts = dataframe[feature].value_counts().to_dict()
            clone_counts = dataframe[feature].value_counts().to_list()
            clone_counts = sorted(clone_counts)
            dup = []
            i = 0
            if(len(set(clone_counts))!=len(value_counts.values())):
                n = len(clone_counts)
                lst = range(0,100)
                while i<n:
                    if i+1<n:
                        if clone_counts[i] == clone_counts[i+1]:
                            dup.append(clone_counts[i])
                            i+=2
                        else: 
                            i+=1
                    else:
                        i+=1
                increase = 0
                for val in value_counts:
                    if(value_counts[val] in dup):
                        value_counts[val] = value_counts[val]+increase
                        increase+=0.1
            self._groups[feature] = value_counts
            dataframe[feature] = dataframe[feature].map(value_counts)
                            
        elif(on_new_col == 'yes' and type(feature) == type(string) or type(feature)==type(integer)):
            value_counts = dataframe[feature].value_counts().to_dict()
            clone_counts = dataframe[feature].value_counts().to_list()
            clone_counts = sorted(clone_counts)
            dup = []
            i = 0
            if(len(set(clone_counts))!=len(value_counts.values())):
                n = len(clone_counts)
                lst = range(0,100)
                while i<n:
                    if i+1<n:
                        if clone_counts[i] == clone_counts[i+1]:
                            dup.append(clone_counts[i])
                            i+=2
                        else: 
                            i+=1
                    else:
                        i+=1
                increase = 0
                for val in value_counts:
                    if(value_counts[val] in dup):
                        value_counts[val]+=increase
                        increase+=0.1
            self._groups[feature] = value_counts
            dataframe[f"{feature}Frequency_encoded"] = dataframe[feature].map(value_counts)
        
        elif(on_new_col == 'no' and type(feature) == list):
            for feature in feature:
                cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")

                value_counts = dataframe[feature].value_counts().to_dict()
                clone_counts = dataframe[feature].value_counts().to_list()
                clone_counts = sorted(clone_counts)
                dup = []
                i = 0
                if(len(set(clone_counts))!=len(value_counts.values())):
                    n = len(clone_counts)
                    lst = range(0,100)
                    while i<n:
                        if i+1<n:
                            if clone_counts[i] == clone_counts[i+1]:
                                dup.append(clone_counts[i])
                                i+=2
                            else: 
                                i+=1
                        else:
                            i+=1
                    increase = 0
                    for val in value_counts:
                        if(value_counts[val] in dup):
                            value_counts[val] = value_counts[val]+increase
                            increase+=0.1
                self._groups[feature] = value_counts
                dataframe[feature] = dataframe[feature].map(value_counts)
                value_counts = {}
                clone_counts = {}
                dup = []
                            
        elif(on_new_col == 'yes' and type(feature) == list):
            for feature in feature:
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
                value_counts = dataframe[feature].value_counts().to_dict()
                clone_counts = dataframe[feature].value_counts().to_list()
                clone_counts = sorted(clone_counts)
                dup = []
                i = 0
                if(len(set(clone_counts))!=len(value_counts.values())):
                    n = len(clone_counts)
                    lst = range(0,100)
                    while i<n:
                        if i+1<n:
                            if clone_counts[i] == clone_counts[i+1]:
                                dup.append(clone_counts[i])
                                i+=2
                            else: 
                                i+=1
                        else:
                            i+=1
                    increase = 0
                    for val in value_counts:
                        if(value_counts[val] in dup):
                            value_counts[val]+=increase
                            increase+=0.1
                self._groups[feature] = value_counts
                dataframe[f"{feature}Frequency_encoded"] = dataframe[feature].map(value_counts)
                value_counts = {}
                clone_counts = {}
                dup = []
                            
        else:
            return error
        
        self.dataframe = dataframe
        return dataframe
    
    def probablity_encoder(self,target,feature,on_new_col = 'no'):
        '''
        Input: pd.DataFrame(),feature,target,on_new_col
        Output: pd.DataFrame(),_groups(Values associated by encoding.)
        Description: Will encode values of feature in datafarme with probablity ratio encoding technique for more context it will be probabltiy of target/1-probablity .
        '''
        if self.dataframe.empty:
            raise TypeError("enter a dataframe in encoder") 
        dataframe = self.dataframe
        string = ''
        error = 'Please Check your parameters again or check documentation if you are not clear about the use.'
        if type(feature) == str:
            cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
            cat_check = len(dataframe[feature].value_counts())/cat_check
            cat_check = cat_check*100
            if cat_check > 20:
                warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
            
        if(on_new_col == 'no' and type(feature) == type(string) or type(feature)==type(integer)):
            prob = dataframe.groupby([feature])[target].mean() # Probablity
            prob_df = pd.DataFrame(prob)
            prob_df['1-prob'] = 1-prob
            prob_df['Ratio'] = prob_df[target]/prob_df['1-prob']
            map_values = prob_df['Ratio'].to_dict()
            self._groups[feature] = map_values 
            dataframe[feature] = dataframe[feature].map(map_values)
            
        elif(on_new_col == 'yes' and type(feature) == type(string) or type(feature)==type(integer)):
            prob = dataframe.groupby([feature])[target].mean() # Probablity
            prob_df = pd.DataFrame(prob)
            prob_df['1-prob'] = 1-prob
            prob_df['Ratio'] = prob_df[target]/prob_df['1-prob']
            map_values = prob_df['Ratio'].to_dict()
            self._groups[feature] = map_values 
            dataframe[f'{feature}Probablity_Encoded'] = dataframe[feature].map(map_values)
            
        elif(on_new_col == 'no' and type(feature) == list):
            for feature in feature:
                cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
                prob = dataframe.groupby([feature])[target].mean() # Probablity
                prob_df = pd.DataFrame(prob)
                prob_df['1-prob'] = 1-prob
                prob_df['Ratio'] = prob_df[target]/prob_df['1-prob']
                map_values = prob_df['Ratio'].to_dict()
                self._groups[feature] = map_values 
                dataframe[feature] = dataframe[feature].map(map_values)
            
        elif(on_new_col == 'yes' and type(feature) == list):
            for feature in feature:
                cat_check = len(dataframe[feature]) - dataframe[feature].isnull().sum()
                cat_check = len(dataframe[feature].value_counts())/cat_check
                cat_check = cat_check*100
                if cat_check > 20:
                    warnings.warn("It looks like features are not categorical or not distributed properly,Do not encode features if not properly distributed or converted to categories, probablity encoding should not be used for encoding continous variables")
                prob = dataframe.groupby([feature])[target].mean() # Probablity
                prob_df = pd.DataFrame(prob)
                prob_df['1-prob'] = 1-prob
                prob_df['Ratio'] = prob_df[target]/prob_df['1-prob']
                map_values = prob_df['Ratio'].to_dict()
                self._groups[feature] = map_values 
                dataframe[f'{feature}Probablity_Encoded'] = dataframe[feature].map(map_values)
        else: return error
        self.dataframe = dataframe
        return dataframe
    