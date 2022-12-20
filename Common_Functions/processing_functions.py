from sklearn.preprocessing import MinMaxScaler
import pandas as pd
    
def data_split(data: pd.DataFrame,
               count_col_name: str = 'priv_count',
               count_thresh: str = 50) -> (pd.DataFrame, pd.DataFrame):
    """
    This function splits the data into model set and future set. Model set is the data used to train, evaluate
    test the model. Future data is data that the model needs to predict on exclusively.
    
    Args:
    data: (pandas: DataFrame) - a pandas data frame with at least 3 columns - "priv_pay_mean", "priv_pay_median" and count_col_name
    count_col_name: (str) - name of the column which is thresholded to make the split
    count_thresh: (int) - threshold value used to split data on count_col_name
    
    Returns:
    model_data (pandas: DataFrame) - data frame with observations that will be used to train and test model
    future_data (pandas: DataFrame) - data frame with all observations on which model will make predictions
    """
    # Filter out rows where the target is invalid
    data = data[(data['priv_pay_median'] > 0) | (data['priv_pay_median'].isnull())]
    # Isolate rows below our specified threshold or null
    future_data = data[(data[count_col_name] <= count_thresh) | (data[count_col_name].isnull())]
    # Isolate rows above our specified threshold
    model_data = data[data[count_col_name] > count_thresh]
    model_data = model_data[model_data.priv_pay_median.notnull()]
    
    return model_data, future_data


def add_unique_identifier(data: pd.DataFrame,
                          id_col_list: list = ['msa','year','site','group']) -> pd.DataFrame:
    """
    Adds a unique identifier to the data which is constructed by joining the entries in the columns of col_list
    with a '_' separator.
    
    Args:
    data (pandas: DataFrame) - a pandas data frame for which a unique identifier needs to be constructed. If using
    default settings it should have at least 4 columns with headers - 'msa', 'year','site' and 'group'   
    id_col_list (list) - a list of columns that uniquely identify all observations in the data frame
    
    Returns:
    data_new (pandas: DataFrame) - a transformed data frame with an additional column for unique identifiers with the
    header 'id'
    """
    data_new = data.copy()
    data_new['id'] = data_new[id_col_list].astype(str).agg('_'.join, axis = 1)
    cols = data_new.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    data_new = data_new[cols]
    
    return data_new


def data_cleaning(data: pd.DataFrame,
                  one_hot: bool = True,
                  dropna: bool = True) -> pd.DataFrame:
    """
    Applies preprocessing to our dataset for ease of use in model building
    
    Args:
    data (pandas: DataFrame) - a pandas data frame which needs to be preprocessed  
    one_hot (bool) - Whether or not categorical variables should be one hot encoded
    dropna (bool) - Whether or not NAs should be pruned
    
    Returns:
    data_new (pandas: DataFrame) - a transformed data frame
    """    
    data_new = data
    if one_hot:
        # One hot encoding
        data_new = pd.get_dummies(data_new)
    if dropna:
        # Drop all NAs across columns
        data_new = data_new.dropna()
    
    return data_new

def hospital_data_agg(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies aggregation to our hospital dataset for merging with our dataset
    
    Args:
    data (pandas: DataFrame) - a pandas data frame which needs to be agrgegated at the MSA level  
    
    Returns:
    hospital_msa (pandas: DataFrame) - an aggregated data frame
    """
    hospital_data = data
    
    # Prepare for aggregated columns
    hospital_data["Is_Teaching"] = (hospital_data["teaching"] == "YES").astype(int)
    hospital_data["Beds_Over_500"] = (hospital_data["beds_grp"] == "500+").astype(int)
    hospital_data["Is_Urban"] = (hospital_data["urban_rural"] == "URBAN").astype(int)
    hospital_data["Is_Private"] = ((hospital_data["ownership"] == "PRIVATE (NOT FOR PROFIT)") |
                                   (hospital_data["ownership"] == "PRIVATE (FOR PROFIT)")).astype(int)
    # Create aggregated columns
    hospital_msa = hospital_data.groupby("MSA_CD").agg({"prvdr_num": "count",
                                                        "Is_Teaching":"mean",
                                                        "Beds_Over_500":"mean",
                                                        "Is_Urban":"mean",
                                                        "Is_Private":"mean"
    })
    hospital_msa.reset_index(inplace=True)
    # Rename newly aggregated columns
    hospital_msa.rename(columns = {"MSA_CD":"msa",
                                   "prvdr_num": "Hospitals",
                                   "Is_Teaching":"PctTeaching",
                                   "Beds_Over_500":"PctLargeHospital",
                                   "Is_Urban":"Urban",
                                   "Is_Private":"PctPrivate"
                                  },
                        inplace=True)
    
    return hospital_msa


# Method from Shruti's code
def standardize_data(train_data: pd.DataFrame,
                     val_data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize training and validation (or test) data using the training data to prepare a MinMaxScaler.
    
    Args:
    train_data (pandas: DataFrame) - a pandas data frame containing our training data  
    val_data (pandas: DataFrame) - a pandas data frame containing our validation (or test) data  
    
    Returns:
    train_data_scaled (pandas: DataFrame) - a pandas data frame containing our scaled training data  
    val_data_scaled (pandas: DataFrame) - a pandas data frame containing our scaled validation (or test) data 
    """
    # Drop variables that do not make sense to standardize (categories and coordinates)
    train_temp = train_data.drop(columns = ['site','cluster','lat','lon'])
    val_temp = val_data.drop(columns = ['site','cluster','lat','lon'])
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Train scaler, scale training data, re-append columns
    train_data_scaled = scaler.fit_transform(train_temp)
    train_data_scaled = pd.DataFrame(train_data_scaled, columns = train_temp.columns)
    train_data_scaled['cluster'] = train_data['cluster'].to_list()
    train_data_scaled['site'] = train_data['site'].to_list()
    train_data_scaled['lat'] = train_data['lat'].to_list()
    train_data_scaled['lon'] = train_data['lon'].to_list()
    
    # Scale validation data, re-append columns
    val_data_scaled = scaler.transform(val_temp)
    val_data_scaled = pd.DataFrame(val_data_scaled, columns = val_temp.columns)
    val_data_scaled['cluster'] = val_data['cluster'].to_list()
    val_data_scaled['site'] = val_data['site'].to_list()
    val_data_scaled['lat'] = val_data['lat'].to_list()
    val_data_scaled['lon'] = val_data['lon'].to_list()
    
    return train_data_scaled, val_data_scaled