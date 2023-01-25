import os
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact

def add_differential_peaks(adata,cluster_basis):
    """
    : Add differential peaks based on clusters using Fisher's exact test
    """
    print("Make sure the input adata has raw counts!")
    adata_copy = adata.copy()
    
    data_bin = pd.DataFrame((adata.X.A>0.0)*1.0,
                            columns=adata.var_names,index=adata.obs_names)
    
    celltypes = adata.obs[cluster_basis]

    communities_to_explore = list(np.unique(celltypes))

    pval_df = pd.DataFrame()
    odds_df = pd.DataFrame()
    col_list = list(data_bin.columns)

    for n, c in enumerate(communities_to_explore):
        print(c)
        pval_lt = list()
        odds_lt = list()

        group1 = pd.DataFrame()
        group2 = pd.DataFrame()

        # Subset one cluster at a time (group1) vs all other cells (group2)
        group1 = data_bin.iloc[np.where(celltypes == c)[0], : ]
        group2 = data_bin.iloc[np.where(celltypes != c)[0], : ]

        for peak in col_list:
            print('\r{}'.format(peak),end="")

            a_ = np.array(group1[peak])
            b_ = np.array(group2[peak])

            pospos = np.sum(a_)
            negpos = len(a_) - pospos
            posneg = np.sum(b_)
            negneg = len(b_) - posneg

            stat_array = np.array([[pospos, negpos], [posneg, negneg]])
            oddsratio, pvalue = fisher_exact(stat_array)

            pval_lt = pval_lt + [pvalue]
            odds_lt = odds_lt + [oddsratio]

        pval_df.loc[:,n] = pval_lt
        odds_df.loc[:,n] = odds_lt

    pval_df.columns = communities_to_explore
    pval_df.index = col_list

    odds_df.columns = communities_to_explore
    odds_df.index = col_list
    
    adata_copy.varm['differential_peaks_odds'] = odds_df
    adata_copy.varm['differential_peaks_pvals'] = pval_df
    
    return adata_copy
    