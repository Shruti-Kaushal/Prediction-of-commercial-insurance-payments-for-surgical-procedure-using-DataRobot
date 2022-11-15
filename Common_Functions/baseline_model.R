library(MLmetrics)

baseline_rdm_forest <- function(data, seed = 123){
  # This function will construct our baseline random forest model.
  # 
  # Args:
  # data (data frame) - a data frame containing our train data
  # seed (integer) - seed for reproducibility
  # 
  # Returns:
  # hospitals_msa (data frame) - a data frame aggregated by MSA
  set.seed(seed) #Set seed for reproducibility
  Random_Forest <- randomForest(
    formula = priv_pay_median ~ .,
    data    = data,
    num.trees = 500,
    mtry = 7,
    nodesize = 20,
    na.action = na.omit,
    importance = TRUE
  )
  return(Random_Forest)
}
make_baseline_prediction <- function(model, dataset){
  # This function will create predictions for our baseline model and add them to the dataset as a new column.
  # 
  # Args:
  # model - Our model to create predictions
  # dataset (data frame) - Our data used for prediction
  # 
  # Returns:
  # data_w_prediction (data frame) - a data frame with a new column for prediction (pred_priv_pay_median)
  data_w_prediction <- dataset %>%
    mutate(pred_priv_pay_median = predict(model, dataset)) %>%
    filter(!is.na(pred_priv_pay_median))
  return(data_w_prediction)
}
get_mape_percentage <- function(dataset_w_pred){
  ret_mape <- mean(abs((dataset_w_pred$priv_pay_median - dataset_w_pred$pred_priv_pay_median)/dataset_w_pred$priv_pay_median),na.rm = T)*100
  return(ret_mape)
}