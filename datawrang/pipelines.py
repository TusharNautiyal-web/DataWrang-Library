import pandas as pd 
import numpy as np
from encoders import Encoder
import warnings
from colorama import Fore
from imputers import Impute
from drop import Dropnan

class Pipeline:
    """
    Input: pipeline inherits encoders, scaling,feature extraction and removal techniques.
    Output: Step wise dataframe manipulation result and a intensive report using pipeline.report
    Description: Will create a pipeline steps where all the implemented functions and its attributes will get assigned 
    Usage: 
    
    dataframe = 'If dataframe directly passed in pipeline no need to add key dataframe with value for eg.
    >>pipe.make(mean_target = {dataframe: pd.DataFrame(),target:'feature-name'})
    
    if dataframe already in pipeline declaration it will take by default same dataframe for each function without need of declaring inside function'
    
    >>pipe = pipeline(dataframe = pd.DataFrame(),target = 'Survived') # You can also pass scaling = True or False to use scaling method by default its StandardScaler.
    >>pipe.make(mean_target = {feature = 'Your-Feature-Name'},label_encoder = {feature = 'Your-Feature-Name'},remove_outliers = {feature})

    target = 'You can insert target value if using encoders in pipeline else leave blank. If target is used no need to pass key target in declaration inside pipeline again for any other encoder'
    Understanding pipe.make():
    pipe.make takes Keyword Arguments it stroes your steps in key value pair so any no of argument can be there but each argument needs a value. hence you can use 
    >>pipe.make(remove_outliers = '') 
    >>pipe.make(remove_outliers = {})
    >>pipe.make(remove_outliers = ())
    >>pipe.make(remove_outliers = 'step4')
    """
    def __init__(self,dataframe = pd.DataFrame(),target = None,scaling = False):
        self.ambig_val = True
        if type(dataframe) != type(pd.DataFrame()):
               raise AttributeError('Error 0x001 wrong type of dataframe is passed in pipeline try again with right pd.DataFrame')
        if dataframe.empty:
            self.ambig_val = False
        
        self.dataframe = dataframe
        self.target = target
        self.report = {"Scaling" : False,
                       "Encoder": [],
                       "encoded-features": [],
                       "imputed-features": [],
                       "scaled-features": [],
                        'Imputer': [],
                       'optimization': False,
                       '_groups': ""
                      }
   # error rectifier to make code reusable and simple for pipeline.    
    def __error_rectifiers(self,dict_value,feature):
        Function_values = dict_value[feature]
        if self.ambig_val == False and 'dataframe' not in Function_values:
            raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
        
        if self.ambig_val == True:
            Function_values['dataframe'] = self.dataframe
        
        elif self.ambig_val == False and 'dataframe' in Function_values:
            if type(Function_values['dataframe'])!=type(pd.DataFrame()):
                raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
            if Function_values['dataframe'].empty:
                raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
            else: self.dataframe =  Function_values['dataframe']
        else:
            return False
        return Function_values
    

    def make(self,**kwargs):
        """
        pipe.make takes Keyword Arguments it stores your steps in key value pair so any no of argument can be there but each argument needs a value. hence you can use 
        >> # First initalize default values.   
        >> pipe = pipeline(dataframe = df,target = 'target-feature')
        >> pipe.make(rand_sample_cat = {'feature': 'feature-name'},
        >>:          frequency_encoder = {'feature': 'feature-name'},
        >>:          endofdist = {'feature': 'feature-name'},
        >>:          label_encoder = {'feature': ['feature-name','feature-name']}....)
        
        >># If default values are not set you can also specify values here.
        >> pipe = pipeline()
        >> pipe.make(rand_sample_cat = {'dataframe': df,'feature': 'feature-name'},
        >>:          frequency_encoder = {'dataframe': df, 'feature': 'feature-name'},
        >>:          endofdist = {'dataframe': df,'feature': 'feature-name'},
        >>:          label_encoder = {'dataframe': df,'feature': ['feature-name','feature-name']}....)
        This flow means 
        step1 = rand_sample_cat : Values, 
        step2 = frequency_encoder : Values and so on ... Do understand you cannot use same function twice in single pipeline.
        This feature of using a same function twice in pipe line will be avalible in future updates of the library.
        """
        # Pipeline objects initalized for each classes.
        encoder = Encoder(self.dataframe)
        drop = Dropnan()
        imputer = Impute()
        
        for val in kwargs:
            if val == 'mean_target':
                mean_target_values = kwargs['mean_target']
                
                if self.target == None:
                    if mean_target_values.get('target',False) == False:
                        raise AttributeError('Error 1x004 target is required feild for mean target encoding please specify target in either pipeline or encoder step.')
                    if type(mean_target_values['target']) != str:
                        raise TypeError('Error 1x005 target is required feild for mean target encoding please check target type in either pipeline or encoder step it should be String.')
                    else: self.target = mean_target_values['target']
                    
                if self.ambig_val == False and 'dataframe' not in mean_target_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    mean_target_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in mean_target_values:
                    if type(mean_target_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if mean_target_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  mean_target_values['dataframe']             
                try:    
                    if(mean_target_values.get('on_new_col',False)== False):
                        mean_target_values['on_new_col'] = 'no'
                        
                    self.dataframe = encoder.mean_target(target = self.target,feature = mean_target_values['feature'], on_new_col =  mean_target_values['on_new_col'])
                    self.report['Encoder'].append('mean_target')
                    self.report['encoded-features'].append(mean_target_values['feature'])
                    self.report['_groups'] = encoder._groups
                    
                except Exception as e:
                    raise AttributeError(e)
                
            elif val == 'label_encoder':
                
                label_encoder_values = kwargs['label_encoder']
                    
                if self.ambig_val == False and 'dataframe' not in label_encoder_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    label_encoder_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in label_encoder_values:
                    if type(label_encoder_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if label_encoder_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  label_encoder_values['dataframe']             
                try:    
                    if(label_encoder_values.get('on_new_col',False)== False):
                        label_encoder_values['on_new_col'] = 'no'
                    self.dataframe = encoder.label_encoder(feature = label_encoder_values['feature'],on_new_col = label_encoder_values['on_new_col'])
                    self.report['Encoder'].append('label_encoder')
                    self.report['encoded-features'].append(label_encoder_values['feature'])
                    self.report['_groups'] = encoder._groups
                    
                except Exception as e:
                    raise AttributeError(e)
                
            elif val == 'frequency_encoder':
                
                frequency_encoder_values = kwargs['frequency_encoder']
                    
                if self.ambig_val == False and 'dataframe' not in frequency_encoder_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    frequency_encoder_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in frequency_encoder_values:
                    if type(frequency_encoder_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if frequency_encoder_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  frequency_encoder_values['dataframe']
                if(frequency_encoder_values.get('on_new_col',False )== False):
                    frequency_encoder_values['on_new_col'] = 'no'
                try:    
                    self.dataframe = encoder.frequency_encoder(feature = frequency_encoder_values['feature'], on_new_col = frequency_encoder_values['on_new_col'])
                    self.report['Encoder'].append('frequency_encoder')
                    self.report['encoded-features'].append(frequency_encoder_values['feature'])
                    self.report['_groups'] = encoder._groups
                    
                except Exception as e:
                    raise AttributeError(e)

            elif val == "probablity_encoder":
                prob_values = kwargs['probablity_encoder']
                
                if self.target == None:
                    if prob_values.get('target',False) == False:
                        raise AttributeError('Error 1x004 target is required feild for mean target encoding please specify target in either pipeline or encoder step.')
                    if type(prob_values['target']) != str:
                        raise TypeError('Error 1x005 target is required feild for mean target encoding please check target type in either pipeline or encoder step it should be String.')
                    else: self.target = prob_values['target']
                    
                if self.ambig_val == False and 'dataframe' not in prob_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    prob_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in prob_values:
                    if type(prob_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if prob_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  prob_values['dataframe']             
                try:    
                    if(prob_values.get('on_new_col',False)== False):
                        prob_values['on_new_col'] = 'no'
                    self.dataframe = encoder.probablity_encoder(target = self.target,feature = prob_values['feature'],on_new_col = prob_values['on_new_col'])
                    self.report['Encoder'].append('probablity_encoder')
                    self.report['encoded-features'].append(prob_values['feature'])
                    self.report['_groups'] = encoder._groups
                    
                except Exception as e:
                    raise AttributeError(e)

            #Imputation Categories.
            elif val == "frequent_category":
                frequent_category_values = kwargs['frequent_category']
                if self.ambig_val == False and 'dataframe' not in frequent_category_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    frequent_category_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in frequent_category_values:
                    if type(frequent_category_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if frequent_category_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  frequent_category_values['dataframe']         
                
                try:
                    self.dataframe = imputer.frequent_category(dataframe = frequent_category_values['dataframe'], features = frequent_category_values['features'])
                    self.report['Imputer'].append('frequent_category')
                    self.report['imputed-features'].append(frequent_category_values['features'])
                
                except Exception as e:
                    raise AttributeError(e)
                

            elif val == "replace_outliers":
                replace_outliers_values = kwargs['replace_outliers']
                if self.ambig_val == False and 'dataframe' not in replace_outliers_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    replace_outliers_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in replace_outliers_values:
                    if type(replace_outliers_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if replace_outliers_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  replace_outliers_values['dataframe']         
                if(replace_outliers_values.get('kind',False)== False):
                    replace_outliers_values['kind'] = 'median'
                    if(replace_outliers_values.get('custom',False) == False): 
                           replace_outliers_values['custom'] = ''
                try:
                   
                    self.dataframe = imputer.replace_outliers(dataframe = replace_outliers_values['dataframe'], features = replace_outliers_values['features'],kind = replace_outliers_values['kind'], custom = replace_outliers_values['custom'])
                    self.report['Imputer'].append('replace_outliers')
                    self.report['imputed-features'].append(replace_outliers_values['features'])
                
                except Exception as e:
                    raise AttributeError(e)
            
            elif val == "remove_outliers":
                remove_outliers_values = kwargs['remove_outliers']
                
                if self.ambig_val == False and 'dataframe' not in remove_outliers_values:
                    raise AttributeError('Error 1x001 dataframe is not present in either encoder or pipeline encoder function requires dataframe.')
                
                if self.ambig_val == True:
                    remove_outliers_values['dataframe'] = self.dataframe
                
                elif self.ambig_val == False and 'dataframe' in remove_outliers_values:
                    if type(remove_outliers_values['dataframe'])!=type(pd.DataFrame()):
                        raise TypeError('Error 1x003 Your Dataframe type is not right please check your dataframe we only support pandas.DataFrame()')
                    if remove_outliers_values['dataframe'].empty:
                        raise AttributeError('Error 1x002 Dataframe is empty please add a proper dataframe for encoder or pipeline')
                    else: self.dataframe =  remove_outliers_values['dataframe']
                try:
                    self.dataframe = imputer.remove_outliers(dataframe = remove_outliers_values['dataframe'], features = remove_outliers_values['features'])
                    self.report['Imputer'].append('remove_outliers')
                    self.report['imputed-features'].append(remove_outliers_values['features'])
                
                except Exception as e:
                    raise AttributeError(e)
            
            elif val == "endofdist":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'endofdist')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.endofdist(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('endofdist')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)

            elif val == "rand_sample":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'rand_sample')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.rand_sample(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('rand_sample')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)

            elif val == "rand_sample_cat":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'rand_sample_cat')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.rand_sample_cat(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('rand_sample_cat')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)

            elif val == "nan_mean":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_mean')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_mean(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('nan_mean')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)
            
            elif val == "nan_median":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_median')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_median(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('nan_median')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)
                        
            elif val == "nan_mode":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_mode')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_mode(dataframe = Function_values['dataframe'],feature = Function_values['feature'])
                    self.report['Imputer'].append('nan_mode')
                    self.report['imputed-features'].append(Function_values['feature'])
                
                except Exception as e:
                    raise AttributeError(e)
            
            elif val == "nan_mean_all":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_mean_all')
                if(Function_values.get('force',False) == False):
                    Function_values['force'] = 'no'
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_mean_all(dataframe = Function_values['dataframe'],force = Function_values['force'])
                    self.report['Imputer'].append('nan_mean_all')
                
                except Exception as e:
                    raise AttributeError(e)
            
            elif val == "nan_mode_all":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_mode_all')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_mode_all(dataframe = Function_values['dataframe'])
                    self.report['Imputer'].append('nan_mode_all')
                
                except Exception as e:
                    raise AttributeError(e)
                    
            elif val == "nan_median_all":
                Function_values = self.__error_rectifiers(dict_value = kwargs,feature = 'nan_median_all')
                if(Function_values == False):
                    raise TypeError('Unknown Error: There is some issues with your input please check it and try again')
                try:
                    self.dataframe = imputer.nan_median_all(dataframe = Function_values['dataframe'])
                    self.report['Imputer'].append('nan_median_all')
                
                except Exception as e:
                    raise AttributeError(e)

            
            else:
                warnings.warn('WMRx002: Please Check if the functions and parameters are correct, or if they are supported in pipeline structure. Some of the functions given are not supported and hence will not work.') 
        return self.report

