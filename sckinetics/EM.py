import os
import time
import scanpy as sc
import pandas as pd
import numpy as np
from math import ceil
from numba import njit

from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
import sklearn.neighbors
from sklearn.decomposition import PCA

from tqdm.auto import tqdm
import contextlib
import joblib
from joblib import Parallel, delayed

from .base import check_is_fitted

class ExpectationMaximization():
    
        
    def __init__(self, maxiter=20, tol=0.005, knn=50, sigma=5.0, sigma_prior = 1.0, threads=20):
        self.maxiter = maxiter
        assert (maxiter >= 15 and maxiter <=30), f"We suggest maxiter between 15 to 30, got: {maxiter}"
        self.tol = tol
        self.knn=knn
        self.sigma=sigma
        self.sigma_prior=sigma_prior
        
        self.threads=threads
        
        self.celltype_priors=None
        self.g=None
        self.alpha_all=pd.DataFrame()
        self.beta_all=pd.DataFrame()
        self.kNN_graph=None

        self.A_ = None
        self.velocities_ = None
        self.curves = None
        
        print("Successfully initiated EM model.")
        print('Number of CPUs: {}.'.format(os.cpu_count()))
        print('Running on {} threads.'.format(threads))
    
    def fit(self, adata, G, celltype_basis=None, select_celltypes=None):
        """
        : adata, G: output from tf_targets
        : celltype_basis: variable name for storing celltypes in adata.obs
        : select_celltypes: a list of cell types to fit the model on; default None (running on all cell types)
        """
    
        def fit_gene_analytic(targets):
 
            maxiter=self.maxiter
            tol=self.tol
            curves = {}
            
            def prior(a,ahat,ascale):
                prior=np.sum(norm.logpdf(a,loc=ahat,scale=ascale))
                return prior

            @njit
            def norm_pdf(x,loc,scale):
                tmp1 = (x-loc)/scale
                tmp2= np.exp(-.5*tmp1*tmp1)/(constant)
                return tmp2
            
            @njit
            def preq(a_s,alpha,beta,sigma,X_latent,den):

                #calculate terms from last step, E
                mean_s = np.dot(a_s,X_latent) #a_s dot x_i for latent cell i

                norm_beta = norm_pdf(beta,loc=mean_s,scale=sigma)
                norm_alpha = norm_pdf(alpha,loc=mean_s,scale=sigma)

                num_L1 = norm_beta - norm_alpha
                num_L2 = beta*norm_beta - alpha*norm_alpha

                L1 = num_L1 / den 
                L2 = -1.0*(num_L2 / den) + 1

                return L1,mean_s

            def marginal_L_faster(a, X_latent,alpha,beta,sigma,ahat):
                mean = np.dot(a,X_latent)
                den = norm.cdf(beta,loc=mean,scale=sigma) - norm.cdf(alpha,loc=mean,scale=sigma)
                marginal_l=np.sum(np.log(den))+prior(a,ahat,sigma_prior)
                return marginal_l,den
            
            @njit
            def calculate_A_min(X_latent):
                outerprod_=np.outer(X_latent[:,0],X_latent[:,0])
                for i_ in np.arange(1, X_latent.shape[1]):
                    outerprod_ = outerprod_ + np.outer(X_latent[:,i_],X_latent[:,i_])
                A_min = -.5 / sigma_prior**2 * np.eye(X_latent.shape[0]) - (.5 * outerprod_ / sigma**2)
                return A_min
            
            #@njit
            def calculate_a_s(mean_s,L1):
                B_min = ahat / (sigma_prior**2) + np.sum(mean_s*X_latent,axis=1)/sigma**2 - np.sum(L1*X_latent,axis=1)/sigma
                a_s = -1.0*np.linalg.pinv(A_min+A_min.T).dot(B_min)
                return a_s

            A = np.zeros((D,D))

            for target in targets:
                target_ = G_[target]
                TForder_ix = target_.transcription_factors_indices
                target_ix = target_.index
                TForder = target_.transcription_factors

                # calculate constants
                X_latent = X_[TForder].values.T
                A_min=calculate_A_min(X_latent)

                #get bounds
                alpha = alpha_[:,target_ix]
                beta = beta_[:,target_ix]

                #EM loop
                ahat = A_[target_ix,TForder_ix]
                a_s = np.copy(ahat)
                marginal_likes = [None]*(maxiter+1)
                marginal_likes[0],den = marginal_L_faster(a_s,X_latent,alpha,beta,sigma,ahat)

                for iteration in np.arange(1, maxiter+1):
                    L1,mean_s = preq(a_s,alpha,beta,sigma,X_latent,den)
                    a_s=calculate_a_s(mean_s,L1)
                    marginal_likes[iteration],den = marginal_L_faster(a_s,X_latent,alpha,beta,sigma,ahat)
                    if iteration>15:
                        if float((marginal_likes[iteration]-marginal_likes[iteration-1])/(marginal_likes[1]-marginal_likes[0]))<tol:
                            break

                A[target_ix,TForder_ix] = a_s.copy()
                curves[target] = marginal_likes

            return A,curves
        
        start = time.time()
        
        celltypes = adata.obs[celltype_basis]
        celltypes_list=list(set(celltypes))
        self.celltypes_list=celltypes_list
        print("Total number of cells: " + str(len(celltypes)))
        print("Cell types:", ', '.join(map(str, celltypes_list)))
        
        ## Declare variables
        As, curves = {}, {}
        velocities_=pd.DataFrame(columns=adata.var_names.to_list())
        
        """
        : Computing or reloading required values
        """

        # compute priors
        celltype_priors, g=self.celltype_priors, self.g
        if not celltype_priors==None and not g.any()==None:
            print("Cell type priors already calculated.")
        else:
            celltype_priors, g = get_prior(adata, G, celltypes)
            self.celltype_priors, self.g = celltype_priors, g 
        
        # get alphas and betas
        alpha_all, beta_all=self.alpha_all, self.beta_all
        if not alpha_all.empty and not beta_all.empty:
            print("Constraints already calculated.")
        else:
            alpha_all, beta_all, kNN_graph = get_constraints(adata, celltype_priors, self.knn, celltypes)
            self.alpha_all, self.beta_all, self.kNN_graph = alpha_all, beta_all, kNN_graph

        """
        : Start of the main for loop to iterate through cell types
        """
        sigma=self.sigma
        sigma_prior=self.sigma_prior
        
        if select_celltypes:
            celltypes_list=select_celltypes
            
        for celltype in celltypes_list:
            print("Running main EM algorithm for for: "+ str(celltype))
            G_ = G[celltype]
            X_ = adata.uns['X_{}'.format(celltype)].astype(np.float64)
            A_ = self.celltype_priors[celltype].values
            alpha_ = alpha_all.loc[X_.index][X_.columns].values
            beta_ = beta_all.loc[X_.index][X_.columns].values
            
            targets=list(G_.keys())
            print("Total number of targets: {}".format(len(targets)))

            #precompute any static values before EM and set parameters
            N1, D = X_.shape
            N2 = 0 #not using steady state
            T1 = -1*(N1+N2)/2.0 * np.log(2*np.pi*sigma**2)
            constant=np.sqrt(2.0*np.pi)*sigma
            
            """
            : STARTING PARALLELIZATION BY BATCHES
            """
            # Divide data in batches
            batch_size = ceil(len(targets) / self.threads)
            batches = [
                targets[ix:ix+batch_size]
                for ix in range(0, len(targets), batch_size)
            ]
            num_batches=len(batches)
            
            with tqdm_joblib(tqdm(desc="Celltype {} on {} batches".format(celltype, num_batches), total=num_batches)) as progress_bar:
                # Divide the work to threads
                parallel_result = Parallel(n_jobs=self.threads)(
                    delayed(fit_gene_analytic)
                    (batch) 
                    for batch in batches
                )

            # Pool returned data from parallel processes
            A = np.sum([parallel_result[i][0] for i in range(num_batches)],axis=0)
            As[celltype]=A
            velocity=np.dot(A, X_.values.T).T
            this_velocity=pd.DataFrame(velocity, index=X_.index.to_list(), columns=X_.columns.to_list(), dtype='float64')
            
            this_curve={}
            for i in range(num_batches):
                this_curve.update(parallel_result[i][1])
            curves[celltype]=this_curve
            
            # filter for TFs that are not targets in the dynamical model
            TFs,targets=np.where(np.sum(A,0)!=0)[0],np.where(np.sum(A,1)!=0)[0]
            invalid_TFs=list(set(TFs)-set(TFs[np.isin(TFs, targets)]))
            this_velocity=this_velocity.drop(columns=X_.columns[invalid_TFs])
            
            velocities_=pd.concat([velocities_, this_velocity], join='outer')

        """
        : Finish iteration. Update all values in the model
        """
        self.A_ = As
        # drop genes that have no calculated velocity across all cells
        self.velocities_= velocities_.reindex(adata.obs_names.to_list()).dropna(how='all', axis='columns')
        self.curves=curves
        
        end = time.time()
        seconds= end - start
        print("Finished ATACVelo calculation in {} seconds.".format(seconds))

        
"""
: HELPER FUNCTIONS
"""
def get_prior(adata, G, celltypes):
    ## compute prior and cosine distances beforehand, this part needs to be cleaned
    #PRIORS
    celltype_priors = {}
    print("Start calculating priors.")
    for celltype in set(celltypes):
        print("Getting priors for: "+str(celltype))
        #read in GRN for that cell type
        G_ = G[celltype]
        X_ = adata.uns['X_{}'.format(celltype)].astype(np.float64)
        
        D = X_.shape[1]

        g = np.zeros((D,D))
        for target in G_.keys():
            target_ = G_[target]
            for tf_ix in target_.transcription_factors_indices:
                g[target_.index,tf_ix] = 1.0

        prior = np.cov(X_.T).astype(np.float64)
        celltype_priors[celltype] = pd.DataFrame(prior*g,columns=list(X_))
    return celltype_priors, g

def get_constraints(adata, celltype_priors, knn, celltypes):

    columns, index = list(adata.var_names), list(adata.obs_names)
    num_cells, num_genes = adata.shape
    data=adata.obsm['X_transformed']
    raw_knn=sklearn.neighbors.kneighbors_graph(adata.obsm['X_pca'],int(knn), mode='connectivity', 
                                   metric='euclidean', n_jobs=-1, include_self=True).toarray()
    # Build kNN on the jaccard matrix
    kNN_graph = sklearn.neighbors.kneighbors_graph(raw_knn, int(knn), mode='distance', 
                                   metric='jaccard', n_jobs=-1, include_self=False).toarray()
    
    print("Start calculating constraints.")
    
    #compute slopes to all neighbors in KNN graph for constraint computation
    prior_genes = {}
    for celltype in set(celltypes):
        prior_genes[celltype] = np.array([columns.index(gene) for gene in list(celltype_priors[celltype])])

    alpha_all,beta_all = np.zeros((num_cells, num_genes), dtype=np.float64), np.zeros((num_cells, num_genes), dtype=np.float64)
    #get velocities for each cell

    for cellrand in range(num_cells):
        print("\r{}".format(cellrand),end="")
        slopedist = data.iloc[np.where(kNN_graph[cellrand,:])[0]].values - data.iloc[cellrand].values
        offset = np.std(slopedist,axis=0) / 10.0
        # clip zero values to avoid nans
        offset = np.clip(offset, 1e-10, np.max(offset))
        #cluster velocities using DBSCAN with cosine distance as metric
        prior = celltype_priors[celltypes[cellrand]]
        prior_pred = prior.dot(data.iloc[cellrand][list(prior)])
        use_genes_ =prior_genes[celltypes[cellrand]]

        e=0.03
        kmeds = DBSCAN(eps=e,metric='cosine',min_samples=2).fit(slopedist)
        while (len(kmeds.components_)>10) and (e>0.01):
            e=e-0.01
            kmeds = DBSCAN(eps=e,metric='cosine',min_samples=2).fit(slopedist)
        while(len(kmeds.components_)<5):
            e=e+0.01
            kmeds = DBSCAN(eps=e,metric='cosine',min_samples=2).fit(slopedist)
        num_components=len(kmeds.components_)
        cosinecorrs = [None]*num_components
        for i in range(num_components):
            cosinecorrs[i]=1.0-cosine(kmeds.components_[i][use_genes_],prior_pred)
        med = kmeds.components_[np.argmax(cosinecorrs)]
        alpha_all[cellrand, :] = med - offset
        beta_all[cellrand, :] = med + offset
    print("\nFinished getting all constraints.")

    return pd.DataFrame(alpha_all.astype(np.float64), columns=columns, index=index), pd.DataFrame(beta_all.astype(np.float64), columns=columns, index=index), kNN_graph
        
"""
: Progress Bar
"""
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()