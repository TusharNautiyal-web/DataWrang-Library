# IMPORTANT:
please update your library to the latest version to use it properly v datawrang == 0.0.1 had bugs and its now fixed in v datawrang == 0.0.3
Data Wrang Library is created For Python for dealing with problems of feature engineering and feature scaling and handling missing values.

# Getting Started Documentaiton

For Installation:
```
!pip install DataWrang
```
<a href = ''>Detailed DataWrang Documentation</a>

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
# A full documentation will be coming soon. Thank you.
```
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
