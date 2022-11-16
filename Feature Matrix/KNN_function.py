#!/usr/bin/env python
# coding: utf-8

# In[3]:


def knn_func(data, neighbours = 3):
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    ## replace all the infinite values with 0's in the data
    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    imputer = KNNImputer(n_neighbors=neighbours)
    
    ## apply the min max scaler to standardize the data
    scaler = MinMaxScaler()
    X_df = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
    imputed_values = imputer.fit_transform(X_df.values)
    imputed_values = pd.DataFrame(scaler.inverse_transform(imputed_values),columns= data.columns)
    return imputed_values

