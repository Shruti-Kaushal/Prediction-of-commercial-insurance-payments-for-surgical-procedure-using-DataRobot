from .processing_functions import standardize_data
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

def knn_func(data: pd.DataFrame,
             neighbours: int = 3) -> pd.DataFrame:
    """
    This function is a helper method defined in order to help determine our optimal number of clusters.
    
    Args:
    data: (pandas: DataFrame) - a pandas data frame to be imputed on
    neighbors: (int) - number of neighbors to use for data imputation
    
    Returns:
    imputed_values (pandas: DataFrame) - data frame after knn imputation has been performed on it
    """
    # Replace all the infinite values with 0's in the data
    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Initialize imputer
    imputer = KNNImputer(n_neighbors=neighbours)
    
    # Apply the min max scaler to standardize the data
    scaler = MinMaxScaler()
    X_df = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
    imputed_values = imputer.fit_transform(X_df.values)
    imputed_values = pd.DataFrame(scaler.inverse_transform(imputed_values),columns= data.columns)
    
    return imputed_values

def optimal_k(data: pd.DataFrame) -> int:
    """
    This function determines the optimal number of clusters, k for KNN imputation.
    We observed that the training data generally gave the optimal number of clusters, so we
    consider only training data in our determination.
    
    Args:
    data: (pandas: DataFrame) - a pandas data frame to be assessed for optimal clustering
    
    Returns:
    (int) - Our optimal number of clusters for KNN imputation
    """
    # Set configuration
    RDM_SEED = 123
    TRAIN_TEST_PROPORTION = 0.8
    
    # Initialize lists
    train_mape = []
    test_mape = []

    # Test various clusterings on a random forest model
    k_list = [2,3,4,5,6,7,8,10,12,14,16]
    for x in k_list:
        model_imputed = knn_func(data,neighbours=x)
        X_input_imp = model_imputed.drop(columns=["priv_pay_median"])
        y_input_imp = model_imputed["priv_pay_median"]

        X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_input_imp,
                                                        y_input_imp,
                                                        train_size = TRAIN_TEST_PROPORTION,
                                                        random_state = RDM_SEED)
        baseline_model_imp = RandomForestRegressor(n_estimators=500, random_state = RDM_SEED)
        baseline_model_imp.fit(X_train_imp, y_train_imp)
        y_train_pred_imp = baseline_model_imp.predict(X_train_imp)
        y_test_pred_imp = baseline_model_imp.predict(X_test_imp)
        train_mape.append(mean_absolute_percentage_error(y_true=y_train_imp, y_pred=y_train_pred_imp))
        test_mape.append(mean_absolute_percentage_error(y_true=y_test_imp, y_pred=y_test_pred_imp))
    
    # Return k that yielded the lowest training MAPE
    return k_list[np.argmin(train_mape)]

def impute_knn(train_data: pd.DataFrame,
               val_data: pd.DataFrame,
               optimal_k: int) -> (pd.DataFrame,pd.DataFrame):
    """
    This function performs knn imputation on our whole dataset, using the number of neighbors specified.
    
    Args:
    train_data: (pandas: DataFrame) - a pandas data frame used to train our imputer. It is then imputed
    val_data: (pandas: DataFrame) - a pandas data frame to be imputed only
    
    Returns:
    train_data_scaled, val_data_scaled (pd.DataFrame,pd.DataFrame) - Our newly imputed datasets
    """
    # Standardize datasets
    train_data_scaled, val_data_scaled = standardize_data(train_data, val_data)
    
    # Initialize imputer, then perform imputation
    knn = KNNImputer(n_neighbors = optimal_k)

    train_data_scaled[list(train_data_scaled.columns)] = knn.fit_transform(train_data_scaled)
    val_data_scaled[list(val_data_scaled.columns)] = knn.transform(val_data_scaled)
    
    return train_data_scaled, val_data_scaled