import pandas as pd
import numpy as np 
from colorama import Fore
import warnings

class Optimizer:
    """
    Input: dataframe
    Output: dataframe,report,selected_dtypes,index
    Formats: 
    int8	i1	8-bit signed integer
    int16	i2	16-bit signed integer
    int32	i4	32-bit signed integer
    int64	i8	64-bit signed integer
    uint8	u1	8-bit unsigned integer
    uint16	u2	16-bit unsigned integer
    uint32	u4	32-bit unsigned integer
    uint64	u8	64-bit unsigned integer
    float16	f2	16-bit floating-point number
    float32	f4	32-bit floating-point number
    float64	f8	64-bit floating-point number
    float128	f16	128-bit floating-point number
    complex64	c8	64-bit complex floating-point number
    complex128	c16	128-bit complex floating-point number
    complex256	c32	256-bit complex floating-point number
    bool	?	Boolean (True or False)
    unicode	U	Unicode string
    object	O	Python objects
    """
    def __init__(self,dataframe = pd.DataFrame()):
        self.dataframe = dataframe
        self.dtypes = dataframe.dtypes.to_dict()
        self.report = None
        self.flag = 0
        self.selected_dtypes = None
        self.index = None
        self.conversion_values = {
                                    'int8': 127,
                                    'int16':32767,
                                    'int32': 2147483648,
                                    'int64': 9223372036854775807,
                                    'uint8': 255,
                                    'uint16': 65535,
                                    'uint32': 4294967295,
                                    'uint64': 18446744073709551615,
                                    'float16': ['10^-4','10^4'],
                                    'float32':['10^-8','10^8'],
                                    'float64':['10^-17','10^17'],
                                    'float128':['10^-20','10^20'],
                                    'object':'text values',
                                    'categories': 'repeated values'
                                        }
        
    def optimize_dataframe(self,index = None):
        """
        Input: index
        Output: dataframe with changed formats or index(optional) for best results..
        Formats supported:
        >>int8 can store integers from -128 to 127.
        >>int16 can store integers from -32768 to 32767.
        >>int64 can store integers from -9223372036854775808 to 9223372036854775807.
        >>uint8	Unsigned integer.	0 to 255
        >>uint16	Unsigned integer.	0 to 65535
        >>uint32	Unsigned integer.	0 to 4294967295
        >>uint64	Unsigned integer	0 to 18446744073709551615
        >>float_	Shorthand for float64.	 
        >>float16	Half precision float.	sign bit, 5 bits exponent, 10 bits mantissa
        >>float32	Single precision float.	sign bit, 8 bits exponent, 23 bits mantissa
        >>float64	Double precision float.	sign bit, 11 bits exponent, 52 bits mantissa
        >>float128	 precise upto 10^20 - 10^-20
        """
        dataframe = self.dataframe
        if(self.dataframe.empty):
            raise AttributeError('Ox101: Error Please check your attributes passed, enter your DataFrame in optimizer to optmize')
        if(index == None):
            warnings.warn("WARNING: indexing helps in increasing your performance you have not passed any index. Your dataframe will still get optimized")
        elif index!=None: 
            dataframe.set_index(index, inplace=True)
        columns = dataframe.columns
        for col in columns:
            if dataframe[col].dtype == 'O':
                pass
        
        return dataframe
    
    def change_dtype(self,feature,dtype):
        # Error Findings.
        if(self.dataframe.empty and type(feature) !=  pd.core.series.Series):
            raise AttributeError('Ox102: Error Please check your attributes passed, enter your DataFrame in optimizer to optmize or enter a valid series. if dont know how to use check documentation. ')
        elif(type(feature) == pd.core.series.Series and self.dataframe.empty):
            self.flag = 1
            warnings.warn('WARNING: you have not passed any dataframe. Your features will not be saved in dataframe please assign the value to dataframe to change the feature type this function will return pandas.series' )
        elif(type(feature) == str and self.dataframe.empty):
            raise AttributeError('Ox103: Incorrect assignment of feature. Feature is not of type pd.series or dataframe is not passed or is empty.Please check and try again')
        
        elif(type(feature) == pd.core.series.Series and not self.dataframe.empty):
            raise AttributeError('Ox107: Incorrect Assignment of feature and dataframe please provide feature in string format or only pass feature pandas.Series or do not use them together ')
        dataframe = self.dataframe
        if(self.flag==1):
            change = np.array(feature)
        else:
            change = np.array(dataframe[feature])
        if(dtype == 'float16'):
            change = change.astype('float16')
        
        elif(dtype == 'float32'):
            change = change.astype('float32')
        
        elif(dtype == 'float64'):
            change = change.astype('float64')
        
        elif(dtype == 'float128'):
            change = change.astype('float128')
        
        elif(dtype == 'int16'):
            change = change.astype('int16')
                  
        elif(dtype == 'int8'):
            change = change.astype('int8')
        
        elif(dtype == 'int32'):
            change = change.astype('int32')
        
        elif(dtype == 'int64'):
            change = change.astype('int64')
            
        elif(dtype == 'Category'):
            change = change.astype('Category')
            
        elif(dtype == 'Object'):
            change = change.astype('Object')
            
        elif(dtype == 'Category'):
            change = change.astype('Category')
        
        elif(dtype == 'uint8'):
            change = change.astype('uint8')
                
        elif(dtype == 'uint16'):
            change = change.astype('uint16')
                
        elif(dtype == 'uint32'):
            change = change.astype('uint32')
                
        elif(dtype == 'uint64'):
            change = change.astype('uint64')
                
            
        else:
            raise TypeError('Ox104: datatype is not correct or cannot convert to such datatype please check documetation for getting more info.')
            
        if self.flag == 1:
            change = pd.Series(change)
            return change
        else:
            dataframe[feature] = change
            self.dataframe = dataframe
            return dataframe