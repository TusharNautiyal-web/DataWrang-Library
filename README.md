<h1 align = 'center' >Welcome To Data Wrang 0.0.4</h1> 
<h2 align = 'center'>A Library Empowering The Data Science Community</h2>
<p align = 'center'>Author: Tushar Nautiyal</p>

<p align="center">
<img  src = 'https://img.shields.io/badge/Version-Alpha-Green.svg'/> <img  src = 'https://img.shields.io/badge/Latest-0.0.4-Green.svg'/> <img src = 'https://img.shields.io/badge/Language-Python-Orange.svg'/>
<img src = 'https://img.shields.io/badge/Older-0.0.3-Green.svg'/>
</p>
## Important
Please update your library to the latest version to use it properly v datawrang == 0.0.3 had bugs and its now fixed in v datawrang == 0.0.4
Data Wrang Library is created For Python for dealing with problems of Feature Engineering and Feature Scaling and handling missing values and other preprocessing and data cleaning problems.
<!-- Place this tag where you want the button to render. -->
<!-- Place this tag in your head or just before your close body tag. -->

# New Feature Coming Soon
1. Update 0.0.5 will be rolling out soon.
2. Update 0.0.5 will be having more encoding method like probablity encoder, binning features and much more.
3. A new way to detect which method of null value imputation you should use for your feature.
4. Update 0.0.6 will be having a feature for faster reads and write of big files.

# Getting Started Documentaiton

For Installation:
```
pip install DataWrang
```
***For Detailed DataWrang <a href = 'https://tusharnautiyal-web.github.io/DataWrang-Library/'>Documentation</a>***

***python.org link<a href = 'https://pypi.org/project/DataWrang/'/>DataWrang - PyPI</a>***
```python
import datawrang
# There are many functions a detailed readme will be avalible soon.

Find_Categorical_dtype_num(df)
# This will return numerical dtype categorical vairables from data frame.
# It should only be used for bigger datasets and should not be used for smaller datasets.        


         
Find_Categorical_return_df(df) 
#return dataframe with categorical variables works for numerical value also.


# For Correlation

Find_corr(dataframe,features,thresh = 12120,sign = ''):
            #thresh = accepts percentage values in decimal
            #sign = accpets +ve , -ve string values.
            #dataframe = pandas dataframe pd.DataFrame()
        # This will return all columns for certain defined +ve or -ve cor-relation.    
#This will find correlation based on +ve -ve sign and thresh please remember to use thresh hold with respect to signs or don't use sign if you are using threshold.

Find_corr_drop(dataframe,features,thresh)
# This Will Delete all the correlation column for certain threshold

# ForRandomSampleimputation 

# Create a impute object and then use it to call the function. There are also other functions like frequenct_category, end_distribution, which will be covered in full documentation.

impute = Impute()
impute.rand_sample(dataframe,feature = "") 

#or

impute.rand_sample_cat(dataframe,feature = "") 
# for categorical you can use rand_sample_cat.


# capture_NAN in new columns feature can be list or feature can be a single string. 
Capture_NaN(df,feature = '')
#.
#.
#.
#.
# A full documentation is out please have a look https://tusharnautiyal-web.github.io/DataWrang-Library. Thank you.
```
## Updates 0.0.4
1. Documentation is updated for the newer version with all new features and usecase mentioned.

'''python

dw.Find_corr_drop_all(dataframe,thresh,feature = '')
#It takes data frame as input with threshold value as thresh in floating format like 90% = 0.9 and will remove all features that are co-related above 90 percent.

encoder = dw.Encoder()
encoder.mean_target(self,dataframe,feature,target, on_new_col = 'no')

'''
## Documentation
Python-code: Go to documentation folder for the code

Main-code: Go to datawrang folder

Website : https://tusharnautiyal-web.github.io/DataWrang-Library/

Credit: Tushar Nautiyal

Email: info@tusharnauityal.ml

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Usage
```python
import datawrang as dw
dw.Find_corr(dataframe,feature)
# import datawrang and then you can use all the functions and classes avalible in the package.
```
For detailed classes and functions visit Documentation. (coming soon)

## Code of Conduct
Visit Code of counduct page to know about usage policies and code of conduct <a href = 'https://github.com/TusharNautiyal-web/DataWrang-Library/blob/main/CODE%20OF%20CONDUCT.md'>Click Here</a>.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
