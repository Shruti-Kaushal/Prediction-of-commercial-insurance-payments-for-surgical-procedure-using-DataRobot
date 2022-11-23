## pass the processed data to this function with catgorical variables One Hot Encoded
## we recognized a pattern where the train and test MAPE attain minima at the same K value, hence we take np.argmin of the train_mape only 
def optimal_k(data):
    RDM_SEED = 123
    TRAIN_TEST_PROPORTION = 0.8
    train_mape = []
    test_mape = []
    # from KNN_function import knn_func
    import knn_func
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_percentage_error

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
    return k_list[np.argmin(train_mape)]

