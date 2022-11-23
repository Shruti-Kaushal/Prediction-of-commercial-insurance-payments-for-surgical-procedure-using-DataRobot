data_split <- function(data, count_col_name = 'priv_count', count_thresh = 50) {
  # This function splits the data into model set and future set. Model set is the data used to 
  # train, evaluate test the model. Future data is what the model needs to predict on.
  # 
  # Args:
  # data (data frame) - a data frame with at least 2 columns - "priv_pay_median" 
  # and count_col_name
  # count_col_name (str) - name of the column which is thresholded to make the split
  # count_thresh (int) - threshold value used to split data on count_col_name
  # 
  # Returns:
  # List of 2 data frames where the first element is model_data and second is future_data
  # model_data (data frame) - data frame with observations that will be used to train and test model
  # future_data (data frame) - data frame with all observations on which model will make predictions 
  
  library(dplyr)
  data <- filter(data, data$priv_pay_median >=0  | is.na(data$priv_pay_mean))
  future_data <- filter(data, data[count_col_name] <= count_thresh | is.na(data[count_col_name]))
  model_data <- filter(data, data[count_col_name] > count_thresh)
  return(list(model_data, future_data))
}

aggregate_hospital_features <- function(hospital_data) {
  # This function will convert our initial hospital data into a version aggregated at
  # the MSA level.
  # 
  # Args:
  # hospital_data (data frame) - a data frame containing the hospital data as it was originally stored
  # 
  # Returns:
  # hospitals_msa (data frame) - a data frame aggregated by MSA
  
  library(dplyr)
  hospitals_msa <- hospital_data %>%
    group_by(MSA_CD) %>%
    summarise(Hospitals = n(),
              PctTeaching = sum(teaching == "YES")/n(),
              PctLargeHospital = sum(beds_grp == "500+")/n(),
              Urban = ifelse(sum(urban_rural == "URBAN")/n() == 1, "Urban","Rural"),
              PctPrivate = sum(ownership == "PRIVATE (NOT FOR PROFIT)" | ownership == "PRIVATE (FOR PROFIT)")/n()) %>%
    rename(msa = MSA_CD)
  return(hospitals_msa)
}
train_test_split <- function(data_to_split, proportion_train = .8, seed = 123){
  # This function will split our dataset into train (dev) and test sets
  # 
  # Args:
  # data_to_split (dta frame) - data to be split into train and test
  # proportion train - proportion of our data to be used in the train set
  # seed - Seed used for reproducible analysis
  # 
  # Returns:
  # List of 2 data frames where the first element is training data and second is test data
  # train (data frame) - train data
  # test (data frame) - test
  
  set.seed(seed) # Set seed for reproducible analysis
  dt = sort(sample(nrow(data_to_split), nrow(data_to_split)*proportion_train)) #Split data
  train <-data_to_split[dt,] #80% training data
  test <-data_to_split[-dt,] #20% test data
  return(list(train, test))
}
