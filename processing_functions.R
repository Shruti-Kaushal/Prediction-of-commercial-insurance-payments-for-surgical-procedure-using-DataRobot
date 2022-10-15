data_split <- function(data, count_col_name = 'priv_count', count_thresh = 50) {
  # This function splits the data into model set and future set. Model set is the data used to 
  # train, evaluate test the model. Future data is what the model needs to predict on.
  # 
  # Args:
  # data (data frame) - a data frame with at least 3 columns - "priv_pay_mean", "priv_pay_median" 
  # and count_col_name
  # count_col_name (str) - name of the column which is thresholded to make the split
  # count_thresh (int) - threshold value used to split data on count_col_name
  # 
  # Returns:
  # List of 2 data frames where the first element is model_data and second is future_data
  # model_data (data frame) - data frame with observations that will be used to train and test model
  # future_data (data frame) - data frame with all observations on which model will make predictions 
  
  library(dplyr)
  data <- filter(data, data$priv_pay_mean >=0 & data$priv_pay_median >= 0)
  future_data <- filter(data, data[count_col_name] <= count_thresh)
  model_data <- filter(data, data[count_col_name] > count_thresh)
  return(list(model_data, future_data))
}


