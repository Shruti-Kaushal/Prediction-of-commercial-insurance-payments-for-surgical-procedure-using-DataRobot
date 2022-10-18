feature_dict = {count_col_name: 'priv_count',\
count_thresh: 50, \
id_col_list: ['msa','year','site','group']}

def data_split(data, count_col_name, count_thresh = 50):
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
    data = data[(data['priv_pay_mean'] >= 0) | (data['priv_pay_mean'].isnull())]
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
    data (pandas: DataFrame) - a transformed data frame with an additional column for unique identifiers with the
    header 'id'
    """
    data['id'] = data[id_col_list].astype(str).agg('_'.join, axis = 1)
    cols = data.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    return data