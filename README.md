<p align = 'center'>
<img src = 'https://user-images.githubusercontent.com/74553737/177100736-64354d6d-00f3-4c54-9235-08f2d4c7a06b.png' width = '200px' height = '100px'/>
</p>

<h1 align = 'center' >Welcome To Data Wrang 0.0.4</h1> 
<h2 align = 'center'>A Library Empowering The Data Science Community</h2>
<p align = 'center'>Author: Tushar Nautiyal</p>

<p align="center">
<img  src = 'https://img.shields.io/badge/Version-Alpha-Green.svg'/> <img  src = 'https://img.shields.io/badge/Latest-0.0.4.2-Green.svg'/> <img src = 'https://img.shields.io/badge/Language-Python-Orange.svg'/>
<img src = 'https://img.shields.io/badge/Older-0.0.3-Green.svg'/>
</p>

## Important

Please update your library to the latest version to use it properly v datawrang == 0.0.3 had bugs and its now fixed in v datawrang == 0.0.4
Data Wrang Library is created For Python for dealing with problems of Feature Engineering and Feature Scaling and handling missing values and other preprocessing and data cleaning problems.
<!-- Place this tag where you want the button to render. -->
<!-- Place this tag in your head or just before your close body tag. -->

# New Feature Coming Soon
1. Datawrang library-repo is updated to version 0.0.5. Start contributing towards library to further modify
2. Datawrang 0.0.5 introduced pipelines for modular pipelining of you preprocessing stream, optimizers for increasing read and load and optimizing your dataframe for big data and at last more encoders and encoding technique with 1 feature or many together option.
3. Issues Fixed: 
    1. Repository with better filing naming and modules.
    2. Attribute Error Issues Fixed Bug Removed.
    3. Better Performance and increased processing times.
    4. Docstrings introduced for each module and its functions.
4. Future updates will include NLP stopwords support for 28+ languages in 0.0.6
5. Also better optimization techniques, auto ml and Series encoder to encode series directly without dataframe.
# Getting Started Documentaiton

For Installation:
```
pip install datawrang
```
***For Detailed DataWrang <a href = 'https://tusharnautiyal-web.github.io/DataWrang-Library/'>Documentation</a>***

***python.org link<a href = 'https://pypi.org/project/DataWrang/'/>DataWrang - PyPI</a>***

```python
import datawrang as dw
# There are many functions a detailed readme will be avalible soon.

dw.Find_Categorical_dtype_num(df)
# This will return numerical dtype categorical vairables from data frame.
# It should only be used for bigger datasets and should not be used for smaller datasets.        


         
dw.Find_Categorical_return_df(df) 
#return dataframe with categorical variables works for numerical value also.


# For Correlation

dw.Find_corr(dataframe = df,features = '',thresh = 0.1 ,sign = ''):
            #thresh = accepts percentage values in decimal
            #sign = accpets +ve , -ve string values.
            #dataframe = pandas dataframe pd.DataFrame()
        # This will return all columns for certain defined +ve or -ve cor-relation.    
#This will find correlation based on +ve -ve sign and thresh please remember to use thresh hold with respect to signs or don't use sign if you are using threshold.

dw.Find_corr_drop(dataframe,features,thresh)
# This Will Delete all the correlation column for certain threshold
# Do understand this is not a drop_all function but a drop function base on feature you passed that means it will drop corr-related columns based on feature given
# For more explanation check the coming up video for use of datawrang.

# ForRandomSampleimputation 

# Create a impute object and then use it to call the function. There are also other functions like frequenct_category, end_distribution, which will be covered in full documentation.

impute = dw.Impute()
impute.rand_sample(dataframe,feature = "") 

#or

impute.rand_sample_cat(dataframe,feature = "") 
# for categorical you can use rand_sample_cat.


# capture_NAN in new columns feature can be list or feature can be a single string. 
dw.Capture_NaN(df,feature = '')
#.
#.
#.
#.
# A full documentation is out please have a look https://datawrang.ml. Thank you.
```
## Updates 0.0.4
1. Documentation is updated for the newer version with all new features and usecase mentioned.

```python

dw.Find_corr_drop_all(dataframe,thresh,feature = '')
#It takes data frame as input with threshold value as thresh in floating format like 90% = 0.9 and will remove all features that are co-related above 90 percent.

encoder = dw.Encoder()
encoder.mean_target(self,dataframe,feature,target, on_new_col = 'no')

```
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
# To acess classes for example Impute
# This is before update 0.0.5
impute = dw.Impute()
dw.impute.nan_mode_all(dataframe = df)

or 

encode = dw.encoder(df)
encode.mean_target(feature = ['<feature-name>'],target = '<Your O/P or Target Feature Name>')
```

# This is after update 0.0.5

from datawrang.encoders import Encoder
encode = Encoder(dataframe = '<-Your Dataframe>')
encode.mean_target(feature = ['<Your Feautres>'],target='Your O/P Feature')
For detailed classes and functions visit Documentation.

from datawrang.imputers import Imputer
impute = Imputer()
impute.nan_mean_all(df)

## Code of Conduct
Visit Code of counduct page to know about usage policies and code of conduct <a href = 'https://github.com/TusharNautiyal-web/DataWrang-Library/blob/main/CODE%20OF%20CONDUCT.md'>Click Here</a>.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
