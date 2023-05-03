import sklearn
import numpy as np
import pandas as pd
import scanpy
from tqdm import tqdm

    
"""
: TF-wide in sillico knockout experiment
: Returns a DataFrame with "perburbation scores"
: note that some of the fields in the returned DataFrame will be na just because some genes were not in the GRN of certain clusters and hence velocity was not calculated
"""
def percell_ablate(model, adata, celltype_basis='cluster'):

    all_cells_diff_df=pd.DataFrame(columns=adata.var_names)

    for celltype in set(adata.obs[celltype_basis]):
        print("Calculating TF ablation on cluster {}".format(celltype))
        X_ = adata.uns['X_{}'.format(celltype)].astype(np.float64)
        A= model.A_[celltype].astype(np.float64)
        # doing this for now need to figure out the nans!
        A=np.nan_to_num(A)
        TFs,targets=np.where(np.sum(A,0)!=0)[0],np.where(np.sum(A,1)!=0)[0]
        invalid_TFs=list(set(TFs)-set(TFs[np.isin(TFs, targets)]))
        valid_gene_idx=list(set(range(0,X_.shape[1]))-set(invalid_TFs))
        original_velocity=A.dot(X_.values.T).T[:,valid_gene_idx]
        diffs = []

        for tf_idx in tqdm(TFs):
            Amut = A.copy()
            # Setting that column value to 0
            Amut[:,tf_idx] = 0
            velocities_mut=Amut.dot(X_.values.T).T[:,valid_gene_idx]
            diff = np.diag(sklearn.metrics.pairwise_distances(original_velocity,velocities_mut,metric='cosine'))
            diffs.append(diff)

        diffs_df=pd.DataFrame(diffs, index=X_.columns[TFs], columns=X_.index).T
        all_cells_diff_df=pd.concat([diffs_df,all_cells_diff_df], join='outer')

    return all_cells_diff_df.dropna(axis=1, how='all').reindex(adata.obs_names.to_list())