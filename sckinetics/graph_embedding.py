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

import seaborn as sns
import matplotlib.pyplot as plt


class VelocityGraph():
    
        
    def __init__(self, model, adata, celltype_basis=None,select_celltypes=None,palette=None, knn=50):
        
        '''
        palette needs to be a set
        select_celltypes: a list of cell types to generate the velocity graph with (only use this option if the model was fit on certain celltypes); default None (running on all cell types)
        '''
        
        if select_celltypes:
            self.adata=adata[adata.obs[celltype_basis].isin(select_celltypes),:]
            self.velocities=model.velocities_.iloc[np.where(adata.obs[celltype_basis].isin(select_celltypes))[0],:]
        else:
            self.adata=adata
            self.velocities=model.velocities_
        print("Make sure that the model had already been fitted!")
        self.knn=knn #for better velocity stream visualizations, we use a smaller knn value to construct the graph
        
        self.graph=None
        self.velocity_graph=None
        self.T=None
        self.backwards_T=None
        
        if palette:
            self.palette=palette
            self.cluster_color=[palette[i] for i in adata.obs['cluster']]
        else:
            self.cluster_color='b'

    def create_velocity_graph(self):
        raw_knn=sklearn.neighbors.kneighbors_graph(self.adata.obsm['X_pca'],int(self.knn), mode='connectivity', 
                                   metric='euclidean', n_jobs=-1, include_self=True).toarray()
        self.graph = sklearn.neighbors.kneighbors_graph(raw_knn, int(self.knn), mode='distance', 
                                       metric='jaccard', n_jobs=-1, include_self=False)   
        N=len(self.adata.obs_names)
        self.velocity_graph = np.zeros((N,N))
        for i in tqdm(range(N)): #for each cell
            neighs_idx = list(self.graph[i].indices)
            # take intersection of the genes that have calculated velocities across all neighbors
            use_genes=self.velocities.iloc[[i]+neighs_idx].dropna(axis='columns', how='any').columns
            data_tmp=self.adata.obsm['X_transformed'][use_genes].values
            dX = data_tmp[neighs_idx,:] - data_tmp[i,:] #compute vectors of cell to neighbors
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
         
    
def embedding_scatter(embedding,V_emb,figsize=(10,10),downsample=False,
                      s=0,c_scatter='g',c_arrows='black', alpha=0.7):
    x = embedding[:, 0]
    y = embedding[:, 1]
    if downsample:
        fig,ax = plt.subplots(figsize=figsize)
        sample_cells = np.random.choice(range(embedding.shape[0]),size=downsample,replace=False)
        ax.quiver(embedding[sample_cells, 0], embedding[sample_cells, 1], 
                   V_emb[sample_cells, 0], V_emb[sample_cells, 1], zorder=1, color=c_arrows)
        ax.scatter(x=x, y=y, s=s,zorder=0, c=c_scatter, alpha=alpha)
        ax.axis("off");
    else:
        fig,ax = plt.subplots(figsize=figsize)
        ax.quiver(embedding[:, 0], embedding[:, 1], 
                   V_emb[:, 0], V_emb[:, 1], zorder=1, color=c_arrows)
        ax.scatter(x=x, y=y, s=s,zorder=0, c=c_scatter, alpha=alpha)
        ax.axis("off");
    return ax 

def embedding_stream(embedding,V_emb,figsize=(10,10),density=2, arrowsize=2, 
                     s=3, linewidth=2, cluster_color='b', stream_color='black', alpha=0.7):
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


def embedding_stream_subset(embedding, adata, celltypes_to_plot, celltype_basis, V_emb,celltypes_colors,plot_alone=True,
                             figsize=(10,10),density=3, arrowsize=2, 
                     s=4.5, linewidth=1.5, stream_color='black', alpha=0.7):
    """
    : plot_alone: True, returns a streamplot with just the specified cluster(s); False, returns a streamplot on the original embedding but velocity vectors only on the specified cluster(s)
    : celltypes_to_plot: a list of celltypes in adata.obs[celltype_basis] to plot stream plot
    : celltypes_colors: a list of color codes organized for each individual cell
    """
    if velocity_embedding is None:
        print("Need to first compute velocity embeddings!")
    if celltypes_colors is None:
        print("Need to first define celltype colors.")
    subset_index=np.where(adata.obs[celltype_basis].isin(celltypes_to_plot))[0]
    cluster_color=pd.Series(celltypes_colors)
    X_grid, V_grid = compute_velocity_on_grid(X_emb=embedding[subset_index], V_emb=V_emb[subset_index],
                       density=density,n_neighbors=50,autoscale=False, adjust_for_stream=True)
    fig,ax = plt.subplots(figsize=figsize)
    ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], 
                   zorder=1, density=density,color=stream_color,
                  arrowsize=arrowsize,linewidth=linewidth)
    if plot_alone==False:
        ax.scatter(x=embedding[:, 0], y=embedding[:, 1], zorder=0, s=s, c=cluster_color, alpha=0.2)
    ax.scatter(x=embedding[subset_index, 0], y=embedding[subset_index, 1], zorder=0, s=s, c=cluster_color[subset_index], alpha=alpha)
    ax.set_title(celltypes_to_plot)
    ax.axis("off");
    return ax

#wrapper function to run all
def plot_velocities_scatter(adata,model,knn=30,embedding_basis=None,figsize=(10,10),
                            downsample=False, s=0,c_scatter='b',c_arrows='black'):
    assert embedding_basis, f"Need to first compute an embedding in scanpy, such as tsne or UMAP."
    embedding=adata.obsm[embedding_basis]

    vg = VelocityGraph(model, adata,knn=knn)
    vg.create_velocity_graph()
    vg.compute_transitions()
    velocity_embedding = vg.embed_graph(embedding)
    ax = embedding_scatter(embedding,velocity_embedding,downsample=downsample,
                           figsize=figsize,s=s,c_scatter=c_scatter,c_arrows=c_arrows)
    return ax,vg

#wrapper function to run all
def plot_velocities_stream(adata,model,knn=30,embedding_basis=None,figsize=(10,10), density=2, 
                           arrowsize=2, s=3, linewidth=2, cluster_color='b', stream_color='black', alpha=0.7):
    assert embedding_basis, f"Need to first compute an embedding in scanpy, such as tsne or UMAP."
    embedding=adata.obsm[embedding_basis]

    vg = VelocityGraph(model, adata, knn=knn)
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