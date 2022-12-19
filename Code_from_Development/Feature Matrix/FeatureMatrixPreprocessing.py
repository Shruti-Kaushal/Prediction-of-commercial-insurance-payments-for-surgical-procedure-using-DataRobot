def preprocess_func():
    
    # Importing modules
    import numpy as np
    import pandas as pd
    
    # Importing the datasets with combined feature, clusters, and mcare_count
    df_raw = pd.read_csv('combined_features.csv')
    df_clusters = pd.read_csv('clusters_only_using_NormCost.csv')
    df_mcare_count = pd.read_csv('priv_mcare_f_pay_2022Oct18.csv')[['msa', 'year', 'site', 'group', 'mcare_count']]
    
    # Left joining them accordingly to have it all in one dataframe
    df_with_clusters = df_raw.merge(df_clusters, how='left', on='group')
    df_with_clusters = df_with_clusters.merge(df_mcare_count, how='left', on=['msa', 'year', 'site', 'group'])
    
    # Dropping the features that are similar to others
    # NOTE: year and group are not dropped since we need them for further experimentation
    df_preprocessed = df_with_clusters.drop(['msa', 'FIPS.State.Code', 'poverty_rate', 'emp', 'ap', 'State',
                                             'priv_pay_mean', 'mcare_pay_mean', 'mcare_pay_sd', 'priv_pay_iqr'],
                                            axis=1)
    
    # One-Hot Encoding / Mapping site values
    # we set site as 1 for impatient and 0 for outpatient or ASC.
    # This helps make sure that the coefficient of impatient are always more than ASC
    # This addresses a part of the of monotonicity constraint
    
    def map_site(val):
        if val == 'Inpatient':
            return 1
        return 0
    
    df_preprocessed['site'] = df_preprocessed['site'].map(lambda x: map_site(x))
    
    # NOTE, this function does not perform:
    #    1. Target encoding for CBSA_NAME
    #    2. k-NN Imputation
    
    return df_preprocessed   