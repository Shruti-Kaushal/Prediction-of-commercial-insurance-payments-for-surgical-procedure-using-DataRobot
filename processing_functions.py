feature_dict = {count_col_name: 'priv_count',\
count_thresh: 50 \
}

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
    data = data[(data.priv_pay_mean >= 0) & (data.priv_pay_median >= 0)]
    future_data = data[data[count_col_name] <= count_thresh]
    model_data = data[data[count_col_name] > count_thresh]
    return model_data, future_data