from .base import check_is_fitted
from .tf_targets import *
from tqdm.auto import tqdm

import scipy
from scipy.sparse import csr_matrix
import sklearn.neighbors

from scvelo.tools.utils import *
from scvelo.tools.velocity_embedding import quiver_autoscale
from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid

import collections.abc # For deprecated Python 3.9 Iterable
collections.Iterable = collections.abc.Iterable
from cellrank.tl.kernels import PrecomputedKernel
from cellrank.tl.estimators import GPCCA
import cellrank as cr

import seaborn as sns
import matplotlib.pyplot as plt


class VelocityGraph():
    
        
    def __init__(self, model, knn=5):
        
        '''
        '''
        
        self.adata=model.adata
        self.model = model #pre-computed velocity model; has to already be fitted
        check_is_fitted(model,'velocities_',"You must pass a fitted model to create the velocity graph. Please fit and solve model first.")
        self.velocities=self.model.velocities_
        self.knn=knn #for better velocity stream visualizations, we use a smaller knn value to construct the graph
        self.graph=None
        self.velocity_graph=None
        self.T=None
        self.backwards_T=None
        
    def create_velocity_graph(self, use_genes=None):
        
#         if use_genes is None:
#             use_genes = range(len(self.model.velocities_.columns))
        raw_knn=sklearn.neighbors.kneighbors_graph(self.adata.obsm['X_pca'],int(self.knn), mode='connectivity', 
                                   metric='euclidean', n_jobs=-1, include_self=True).toarray()
        self.graph = sklearn.neighbors.kneighbors_graph(raw_knn, int(self.knn), mode='distance', 
                                       metric='jaccard', n_jobs=-1, include_self=False)   
        N=len(self.adata.obs_names)
        self.velocity_graph = np.zeros((N,N))
        for i in tqdm(range(N)): #for each cell
            neighs_idx = list(self.graph[i].indices)
            use_genes=self.velocities.iloc[[i]+neighs_idx].dropna(axis='columns', how='any').columns
            data_tmp=self.adata.obsm['X_transformed'][use_genes].values
            dX = data_tmp[neighs_idx,:] - data_tmp[i,:] #compute vectors of cell to neighbors
            #self.velocity_graph[i,neighs_idx] = cosine_correlation(dX[:,use_genes], self.velocities[i][use_genes]) 
            self.velocity_graph[i,neighs_idx] = cosine_custom_correlation(dX, self.velocities[use_genes].values[i]) 
        self.velocity_graph_neg = np.clip(self.velocity_graph, -1, 0)
        self.velocity_graph = np.clip(self.velocity_graph, 0, 1)
        
        
    def compute_transitions(self,scale=10,negative=True):
        check_is_fitted(self,'velocity_graph',"Compute velocity graph first!")
        T = csr_matrix(np.expm1(self.velocity_graph * scale)).astype(np.longdouble) #gaussian kernel
        
        if negative:
            T += csr_matrix(np.expm1(self.velocity_graph_neg * scale)).astype(np.longdouble)
        T.data += 1.0
        T.setdiag(0)
        self.T = normalize(T)
        self.T.eliminate_zeros()
        
        self.backwards_T=normalize(T.T)
        self.backwards_T.eliminate_zeros()
        
    def embed_graph(self,embedding):
        V_emb = np.zeros(embedding.shape) #code from scVelo
        for i in range(self.T.shape[0]):
            indices = self.T[i].indices
            dX = embedding[indices] - embedding[i, None]  #vectors of nearest neighbors
            dX /= norm(dX)[:, None]
            dX[np.isnan(dX)] = 0
            probs = self.T[i].data #transition probabilities
            V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)
        return V_emb
    
    # If user wants to use cellrank
    def run_cellrank(self):
#         self.adata.uns['transition matrix']=self.T
#         cr.tl.initial_states(self.adata, cluster_key="cluster",key='transition matrix',show_progress_bar=False)
#         cr.pl.initial_states(self.adata, discrete=True)
        kernel = PrecomputedKernel(self.T.toarray(),adata=self.adata)
        g = GPCCA(kernel)
        g.compute_schur(method='brandts')
        g.compute_macrostates()
        g.compute_terminal_states()
        return g
         
        
def embedding_scatter(embedding,V_emb,figsize=(10,10),downsample=False,
                      s=3,c_scatter='grey',c_arrows='r'):
    x = embedding[:, 0]
    y = embedding[:, 1]
    if downsample:
        fig,ax = plt.subplots(figsize=figsize)
        sample_cells = np.random.choice(range(embedding.shape[0]),size=downsample,replace=False)
        ax.quiver(embedding[sample_cells, 0], embedding[sample_cells, 1], 
                   V_emb[sample_cells, 0], V_emb[sample_cells, 1], zorder=1)
        ax.scatter(x=x, y=y, zorder=0, c=c_scatter)
        ax.axis("off");
    else:
        fig,ax = plt.subplots(figsize=figsize)
        ax.quiver(embedding[:, 0], embedding[:, 1], 
                   V_emb[:, 0], V_emb[:, 1], zorder=1)
        ax.scatter(x=x, y=y, zorder=0)
        ax.axis("off");
    return ax 

def embedding_stream(embedding,V_emb,figsize=(10,10),density=2, arrowsize=2, 
                     s=3, linewidth=2, cluster_color='grey', stream_color='black', alpha=0.7):
    x = embedding[:, 0]
    y = embedding[:, 1]
    X_grid, V_grid = compute_velocity_on_grid(X_emb=embedding, V_emb=V_emb,
                       density=density,n_neighbors=30,autoscale=False, adjust_for_stream=True)
    fig,ax = plt.subplots(figsize=figsize)
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], 
                   zorder=1, density=density,color=stream_color,#color=V_grid[1],
                  arrowsize=arrowsize,linewidth=linewidth)#,cmap='hot')
    ax.scatter(x=embedding[:, 0], y=embedding[:, 1], zorder=0, s=s, c=cluster_color, alpha=alpha)
    ax.axis("off");
    return ax
        
#wrapper function to run all
def plot_velocities_scatter(adata,model,embedding_basis=None,figsize=(10,10),
                            downsample=False, s=3,c_scatter='grey',c_arrows='r'):
    assert embedding_basis, f"Need to first compute an embedding in scanpy, such as tsne or UMAP."
    embedding=adata.obsm[embedding_basis]
        
    vg = VelocityGraph(adata,model)
    vg.create_velocity_graph()
    vg.compute_transitions()
    velocity_embedding = vg.embed_graph(embedding)
    ax = embedding_scatter(embedding,velocity_embedding,downsample=downsample,
                           figsize=figsize,s=s,c_scatter=c_scatter,c_arrows=c_arrows)
    return ax,vg
    
#wrapper function to run all
def plot_velocities_stream(adata,model,embedding_basis=None,figsize=(10,10), density=2, 
                           arrowsize=2, s=3, linewidth=2, cluster_color='grey', stream_color='black', alpha=0.7):
    assert embedding_basis, f"Need to first compute an embedding in scanpy, such as tsne or UMAP."
    embedding=adata.obsm[embedding_basis]
        
    vg = VelocityGraph(adata,model)
    vg.create_velocity_graph()
    vg.compute_transitions()
    velocity_embedding = vg.embed_graph(embedding)
    ax = embedding_stream(embedding,velocity_embedding,figsize=figsize, density=density, 
                          arrowsize=arrowsize, s=s, linewidth=linewidth, cluster_color=cluster_color, 
                          stream_color=stream_color, alpha=alpha)
    return ax,vg

def cosine_correlation(dX, Vi): #function taken directly from scVelo
        #dX -= dX.mean(-1)[:, None]
        Vi_norm = vector_norm(Vi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.zeros(dX.shape[0]) if Vi_norm == 0 else np.einsum('ij, j', dX, Vi) / (norm(dX) * Vi_norm)[None, :]
           # result = 1.0 - scipy.spatial.distance.cosine(dX,Vi)
        return result
    
def cosine_custom_correlation(dX, Vi): #function taken directly from scVelo
        result = []
        for i in range(dX.shape[0]):
            result.append(1.0-scipy.spatial.distance.cosine(dX[i,:],Vi))
        return np.array(result)
    
def get_iterative_indices(indices, index, n_recurse_neighbors=2, max_neighs=None):
    
    def iterate_indices(indices, index, n_recurse_neighbors):
        return indices[iterate_indices(indices, index, n_recurse_neighbors - 1)] \
            if n_recurse_neighbors > 1 else indices[index]
    
    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices