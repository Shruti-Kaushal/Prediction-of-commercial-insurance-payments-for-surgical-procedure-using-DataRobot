def data_split(data, count_col_name = 'priv_count', count_thresh = 50):
    """
    This function splits the data into model set and future set. Model set is the data used to train, evaluate
    test the model. Future data is what the model needs to predict on.
    
    Args:
    data (pandas: DataFrame) - a pandas data frame with at least 3 columns - "priv_pay_mean", "priv_pay_median" and count_col_name
    count_col_name (str) - name of the column which is thresholded to make the split
    count_thresh (int) - threshold value used to split data on count_col_name
    
    Returns:
    model_data (pandas: DataFrame) - data frame with observations that will be used to train and test model
    future_data (pandas: DataFrame) - data frame with all observations on which model will make predictions
    """
    data = data[(data['priv_pay_mean'] > 0) | (data['priv_pay_mean'].notnull())]
    future_data = data[(data[count_col_name] <= count_thresh) | (data[count_col_name].isnull())]
    model_data = data[data[count_col_name] > count_thresh]
    return model_data, future_data


def add_unique_identifier(data, id_col_list=['msa','year','site','group']):
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


def data_cleaning(data, one_hot = True, dropna = True):
    """
    Applies preprocessing to our dataset for ease of use in model building
    
    Args:
    data (pandas: DataFrame) - a pandas data frame which needs to be preprocessed  
    one_hot (bool) - Whether or not categorical variables should be one hot encoded
    dropna (bool) - Whether or not NAs should be pruned
    
    Returns:
    data_new (pandas: DataFrame) - a transformed data frame
    """
    import pandas as pd
    
    data_new = data
    if one_hot:
        data_new = pd.get_dummies(data_new)
    if dropna:
        data_new = data_new.dropna()
    return data_new

def hospital_data_agg(data):
    """
    Applies aggregation to our hospital dataset for merging with our dataset
    
    Args:
    data (pandas: DataFrame) - a pandas data frame which needs to be agrgegated at the MSA level  
    
    Returns:
    hospital_msa (pandas: DataFrame) - an aggregated data frame
    """
    
    import pandas as pd
    
    hospital_data = data
    
    hospital_data["Is_Teaching"] = (hospital_data["teaching"] == "YES").astype(int)
    hospital_data["Beds_Over_500"] = (hospital_data["beds_grp"] == "500+").astype(int)
    hospital_data["Is_Urban"] = (hospital_data["urban_rural"] == "URBAN").astype(int)
    hospital_data["Is_Private"] = ((hospital_data["ownership"] == "PRIVATE (NOT FOR PROFIT)") |
                                   (hospital_data["ownership"] == "PRIVATE (FOR PROFIT)")).astype(int)


    hospital_msa = hospital_data.groupby("MSA_CD").agg({"prvdr_num": "count",
                                                        "Is_Teaching":"mean",
                                                        "Beds_Over_500":"mean",
                                                        "Is_Urban":"mean",
                                                        "Is_Private":"mean"
    })
    hospital_msa.reset_index(inplace=True)
    hospital_msa.rename(columns = {"MSA_CD":"msa",
                                   "prvdr_num": "Hospitals",
                                   "Is_Teaching":"PctTeaching",
                                   "Beds_Over_500":"PctLargeHospital",
                                   "Is_Urban":"Urban",
                                   "Is_Private":"PctPrivate"
                                  },
                        inplace=True)
    return hospital_msa