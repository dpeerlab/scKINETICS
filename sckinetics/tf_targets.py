import os
import numpy as np
import pandas as pd

import MOODS.parsers
import MOODS.tools
import MOODS.scan
import tempfile

from Bio.SeqUtils import GC
from ucsc_genomes_downloader import Genome
from gimmemotifs.motif import read_motifs
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
import mygene
import scanpy

from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from math import ceil
from tqdm.auto import tqdm
import contextlib
import joblib
from joblib import Parallel, delayed

class SeqRecord():
    def __init__(self,seqid,seq):
        self.id = seqid
        self.seq = seq
        self.GC = GC(seq)
        
class TargetRecord():
    def __init__(self,gene_name,index=None):
        self.gene_name = gene_name
        self.transcription_factors = []
        self.index = index
        self.transcription_factors_indices = []     
    def add_transcription_factor(self,gene_name,index=None):
        self.transcription_factors.append(gene_name)
        self.transcription_factors_indices.append(index)
    def update(self, target_record):
        self.transcription_factors = self.transcription_factors + target_record.transcription_factors
        self.transcription_factors_indices = self.transcription_factors_indices + target_record.transcription_factors_indices  
        
class GenomeRecord():
    def __init__(self, genome, chromosomes=None):
        
        if genome == 'mm10': #todo: pull these automatically            
            print("Loading genome (this make take a while!) ...")
            self.genome = Genome(assembly='mm10',chromosomes=chromosomes)     
            
            print("Loading motifs ...")
            #todo: replace path with variable as in celloracle
            self.motifs = read_motif_file("CisBP_ver2_Mus_musculus.pfm")
            
            TXDB= importr('TxDb.Mmusculus.UCSC.mm10.knownGene')
            self.genome_annotations = TXDB.TxDb_Mmusculus_UCSC_mm10_knownGene
            self.species = 'mouse'

        
        elif genome == 'hg38':
            print("Loading genome (this make take a while!) ...")
            self.genome = Genome(assembly='hg38',chromosomes=chromosomes)
            #todo: replace path with variable as in celloracle
            print("Loading motifs ...")
            self.motifs = read_motif_file("CisBP_ver2_Homo_sapiens.pfm")
            
            TXDB= importr('TxDb.Hsapiens.UCSC.hg38.knownGene')
            self.genome_annotations = TXDB.TxDb_Hsapiens_UCSC_hg38_knownGene
            self.species = 'human'
            

class PeakAnnotation():
    
    def __init__(self, adata, genome = 'mm10', chromosomes = None):
        
        print("Starting peak annotation. Make sure the X matrix in adata has been log transformed.")
        adata.var_names=[x.upper() for x in adata.var_names]
        self.genes = adata.var_names
        self.adata = adata
        
#         #set up R environment
#         print("Checking R packages are installed ...")
#         packnames = ("BiocManager")
#         bioc_packnames =('BiocGenerics','S4Vectors','methods',
#                          'GenomicFeatures','GenomicRanges','IRanges','TxDb.Hsapiens.UCSC.hg38.knownGene',
#                          'TxDb.Mmusculus.UCSC.mm10.knownGene','ChIPseeker')

#         utils = rpackages.importr('utils')
#         utils.chooseCRANmirror(ind=1) 

#         names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
#         if len(names_to_install) > 0:
#             utils.install_packages(StrVector(names_to_install))

#         bioc =  importr("BiocManager")
#         names_to_install = [x for x in bioc_packnames if not rpackages.isinstalled(x)]
#         if len(names_to_install) > 0:
#             bioc.install(StrVector(names_to_install))
        
#         print(" ")
#         print("Finished setting up R packages")

        # loading genome-specific files needed for motif calling & target identification
        if type(genome) == GenomeRecord:
            self.genome = genome.genome
            self.motifs = genome.motifs
            self.genome_annotations = genome.genome_annotations
        
        elif type(genome) == str:
            
            if genome == 'mm10': #todo: pull these automatically            
                print("Loading genome (this make take a while!) ...")
                self.genome = Genome(assembly='mm10',chromosomes=chromosomes)     
                print("Loading motifs ...")
                #todo: replace path with variable as in celloracle
                #self.motifs = read_motifs("atacvelo/data/motif_data/CisBP_ver2_Mus_musculus.pfm")
                self.motifs = read_motif_file("CisBP_ver2_Mus_musculus.pfm",self.genes)

                TXDB= importr('TxDb.Mmusculus.UCSC.mm10.knownGene')
                self.genome_annotations = TXDB.TxDb_Mmusculus_UCSC_mm10_knownGene
                self.species = 'mouse'


            elif genome == 'hg38':
                print("Loading genome (this make take a while!) ...")
                self.genome = Genome(assembly='hg38',chromosomes=chromosomes)
                #todo: replace path with variable as in celloracle
                print("Loading motifs ...")
                self.motifs = read_motif_file("CisBP_ver2_Homo_sapiens.pfm",self.genes)

                TXDB= importr('TxDb.Hsapiens.UCSC.hg38.knownGene')
                self.genome_annotations = TXDB.TxDb_Hsapiens_UCSC_hg38_knownGene
                self.species = 'human'
            

    #wrapper to run all of the below functions sequentially        
    def call_motifs(self,peak_bed_file, pvalue=1e-3,filterGenes=True,max_upstream_distance=500, max_downstream_distance=3000):
        print("Calling motifs with default settings...")
        targets = self.annotate_peaks(peak_bed_file,filterGenes=filterGenes,
                                      max_upstream_distance=max_upstream_distance, 
                                      max_downstream_distance=max_downstream_distance)
        
        reduced_peak_bed_file = targets[['seqnames','start','end']].drop_duplicates()
        reduced_peak_bed_file.columns = ['chrom','chromStart','chromEnd']
        record_dict = self.extract_peak_sequence(reduced_peak_bed_file)
        self.genome.delete()
        bgs = self.compute_background(record_dict)

        motifs = self.scan_peaks(record_dict,bgs,pvalue=pvalue,pseudocount=0.01,filterGenes=filterGenes)
        
        print("Formatting pairs... (this might take a while!)")
        self.pairs = motifs.merge(targets,on=['seqnames','start','end'])
        tmp = []
        for peak in range(self.pairs.shape[0]):
            peak = self.pairs.iloc[peak,:3].values.astype(np.str)
            tmp.append("{}:{}-{}".format(peak[0],peak[1],peak[2]))
        self.pairs['peak_name'] = tmp
        self.pairs = self.pairs.set_index("peak_name")
        
        print("\nFinished motif calling.")
        self.targets = targets
        self.motifs = motifs
    
    def prepare_data(self, scaling='softplus', imputed_key='X_magic'):

        scanpy.pp.pca(self.adata, n_comps=50)
        if imputed_key not in self.adata.obsm.keys():
            print("Starting MAGIC imputation")
            adata=run_MAGIC(self.adata, use_rep = 'X_pca')
            temp_data=self.adata.obsm['X_magic']
        else:
            temp_data=self.adata.obsm[imputed_key]
        
        if type(temp_data)!=np.ndarray:
            temp_data=temp_data.toarray()
        #data needs to be strictly positive for ODE model
        #using softplus
        if scaling == 'softplus':
            self.adata.obsm['X_transformed'] = pd.DataFrame(np.log(1.0+np.exp(temp_data)),columns=self.adata.var_names,
                         index=self.adata.obs_names)
        elif scaling == "linear":    
            transformed=temp_data - np.min(temp_data.flatten()) + .01
            self.adata.obsm['X_transformed'] = pd.DataFrame(transformed,columns=self.adata.var_names,
                         index=self.adata.obs_names)

    
    #after running call_motifs, creates target record objects optionally for a specific set of peaks
    #e.g. cell -type specific peaks
    def prepare_target_annotations(self,peaks=None,cluster_key="cluster",cluster=None,imputed_basis='X_magic'):
        
        self.prepare_data(imputed_key=imputed_basis)
        X =self.adata.obsm['X_transformed']

        if cluster is not None:
            X = X.iloc[np.where(self.adata.obs[cluster_key]==cluster)[0]]
        
        if peaks is not None:
            pairs = self.pairs.loc[peaks]
        else:
            pairs = self.pairs
            
        pairs = pairs.drop_duplicates(['TF','target'])
        
        tf_targets_both = np.union1d(pairs['TF'],pairs['target'])
        
        genes_model = np.intersect1d(tf_targets_both,self.genes)
        gene_index_target = pd.DataFrame(genes_model)
        gene_index_target['target_index'] = range(gene_index_target.shape[0])
        pairs = pairs.merge(gene_index_target,left_on='target',right_on=0).drop(0,axis=1)

        gene_index_TF = pd.DataFrame(genes_model)
        gene_index_TF['TF_index'] = range(gene_index_TF.shape[0])
        pairs = pairs.merge(gene_index_TF,left_on='TF',right_on=0).drop(0,axis=1)
        
        G = {}
        for record in pairs.iterrows():
            target = TargetRecord(record[1]['target'],index=record[1]['target_index'])
            target.add_transcription_factor(record[1]["TF"],index=record[1]['TF_index'])
            try: #if this target was already created
                G[record[1]['target']].update(target)
            except: #if first instance of target
                G[record[1]['target']] = target
            
        self.adata.uns['X_{}'.format(cluster)] = X[genes_model] #filtered for cell type (optional) and TFs targets
        
        return G

    def extract_peak_sequence(self, peak_bed_file):
        print("Extracting peak sequence...")
        record_dict={}
        for i,row in enumerate(peak_bed_file.iterrows()):
            print("\r{}".format(i),end="")
            peak = row[1].values
            peakname="_".join(peak.astype(np.str))
            record_dict[peakname] = SeqRecord(peakname,self.genome._chromosomes[peak[0]][peak[1]:peak[2]])
        return record_dict

    def compute_background(self,record_dict):
        print("\nComputing background...")
        GC_content = np.array([record_dict[rec].GC for rec in record_dict])
        peak_names = np.array([record_dict[rec].id for rec in record_dict])
        
        self.quantiles = np.array(pd.qcut(GC_content,[0, .25, .5, .75, 1.],
                                     labels=['first','second','third','fourth']))
 
        bgs = []
        for quantile in ['first','second','third','fourth']:
            seqs = peak_names[np.where(self.quantiles==quantile)[0]]
            all_seqs = ""
            for seq in tqdm(seqs):
                all_seqs = all_seqs + str(record_dict[seq].seq)
            bgs.append(MOODS.tools.bg_from_sequence_dna(all_seqs,1))
        return bgs
            
    def scan_peaks(self,record_dict, bgs, window_size=7,pseudocount=1,pvalue=1e-3, filterGenes=True):
        
        print("Scanning peaks...")
        
        #CellRanger code
        def _pwm_to_moods_matrix(pwm, bg, pseudocount):
            """Convert JASPAR motif into a MOODS log_odds matrix, using the give background distribution & pseudocounts
            """
            with tempfile.NamedTemporaryFile() as fn:
                f = open(fn.name, "w")
                for base in range(4):
                    line = " ".join(str(x) for x in pwm[base])
                    f.write(line + "\n")

                f.close()

                return MOODS.parsers.pfm_to_log_odds(fn.name, bg, pseudocount)
        
        all_reqs = []
        peak_names = np.array([record_dict[rec].id for rec in record_dict])
        columns = ['seqnames','start','end','TF','score']#,'strand',
           #'motif_chr','motif_start','motif_stop','matched_seq','motif_consensus','GC_quantile']
        genes=set(self.genes)
        
        for q,quantile in enumerate(['first','second','third','fourth']):
            bg = bgs[q]
            matrices = [_pwm_to_moods_matrix(motif.pwm, bg, pseudocount) for motif in self.motifs]
            matrices = matrices + [MOODS.tools.reverse_complement(m) for m in matrices]
            thresholds = [MOODS.tools.threshold_from_p(m, bg, pvalue) for m in matrices]

            scanner = MOODS.scan.Scanner(window_size)
            scanner.set_motifs(matrices, bg, thresholds)

            seqs = peak_names[np.where(self.quantiles==quantile)[0]]
            
            for i,seq in enumerate(seqs): 
                print("\r Peak {} in {} quantile".format(i, quantile),end="")
                chr_,start_,stop_ = seq.split("_")
                start_ = int(start_)
                stop_ = int(stop_)

                moods_scan_res = scanner.scan(str(record_dict[seq].seq)) 

                for (motif_idx, hits) in enumerate(moods_scan_res):
                    motif = self.motifs[motif_idx % len(self.motifs)]
                    # strand = "-" if motif_idx >= len(self.motifs) else "+"
                    scores = [h.score for h in hits]
                    if len(hits)>0:
                        h = hits[np.argmax(scores)]
                        # motif_start = start_ + int(h.pos) + 1
                        # motif_end = start_ + int(h.pos) + len(motif) + 1
                        score = round(h.score, 4)
                        #if score > 4:
                        motif_names = motif.factors['included']#motif.factors['direct'] + motif.factors['indirect\nor predicted']
                        for motif_name in motif_names:#np.intersect1d(motif_names,self.genes):
                            record = [chr_, start_, stop_, motif_name, score]#, strand, chr_, motif_start, motif_end,
                                  #str(record_dict[seq].seq)[h.pos:(h.pos+len(motif))],str(motif.consensus),quantile]
                            all_reqs.append(record)
        
        print("Finished peak scanning. Compiling all data (this might take a while!!)\n")
        df = pd.DataFrame(all_reqs, columns = columns)
        if filterGenes:
            print("Filtering genes...")
            df = df[df.TF.isin(genes)]
            df.reset_index(drop=True, inplace=True)
        df['start'] = df['start'].values.astype(np.int32)
        df['end'] = df['end'].values.astype(np.int32)
 
        return df.sort_values("score")[::-1].drop_duplicates(['seqnames','start','end','TF'])
    
    def annotate_peaks(self,peak_bed_file,max_upstream_distance=500, max_downstream_distance=3000,filterGenes=True, differential_peaks=True, whitelist=None):
        print("Annotating peaks")
        base = importr('base')
        GF = importr('GenomicFeatures')
        GR = importr('GenomicRanges')
        BG = importr("BiocGenerics")
        IR = importr('IRanges')
        S4V = importr('S4Vectors')
        CSK = importr('ChIPseeker')
        meth = importr('methods')
        
        pandas2ri.activate() 
        print("Running ChIPSeeker ...")       
        peaks = GR.GRanges(
        seqnames = S4V.Rle(peak_bed_file['chrom']),
        ranges = IR.IRanges(peak_bed_file['chromStart'],peak_bed_file['chromEnd']))
        
        r_df_annoPeakTab = base.as_data_frame(meth.slot(CSK.annotatePeak(peaks,TxDb = self.genome_annotations),"anno"))
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            annoPeakTab = ro.conversion.rpy2py(r_df_annoPeakTab)
       
        index = (annoPeakTab['distanceToTSS'] > (-1 * max_downstream_distance)) & (annoPeakTab['distanceToTSS'] <= max_upstream_distance)
        annoPeakTab = annoPeakTab.loc[index]

        print("Converting gene annotations ...")          
        
        mg = mygene.MyGeneInfo()
#         gene_names=[]
#         num_failed=0
#         for i in annoPeakTab['geneId']:
#             try:
#                 gene_names.append(mg.getgene(i)['symbol'].upper())
#             except:
#                 gene_names.append('failed')
#                 num_failed+=1

        queries = mg.querymany(annoPeakTab['geneId'].values,species=self.species)
        gene_names = []
        num_failed=0
        for gene in queries:
            try:
                gene_names.append(gene['symbol'])
            except:
                gene_names.append("NA")
                num_failed+=1

        annoPeakTab['target'] = [gene.upper() for gene in gene_names]
        if (num_failed/len(gene_names))>0.005:
            print("Gene matching to database failed. Data not usable")
        else:
            annoPeakTab=annoPeakTab[annoPeakTab['target']!='failed']

        if filterGenes:
            print("Filtering genes ...")
            annoPeakTab = annoPeakTab[annoPeakTab.target.isin(self.genes)]
            annoPeakTab.reset_index(drop=True, inplace=True)
        
        
        return annoPeakTab
          
        
def read_peak_bed(peak_bed_file_path,npeaks=10):
    peak_bed_file = pd.read_csv(peak_bed_file,sep='\t',header=None).iloc[:npeaks,:3]
    peak_bed_file.columns = ['chrom','chromStart','chromEnd']
    peak_bed_file['chromStart'] = peak_bed_file['chromStart'].values.astype(np.int32)
    peak_bed_file['chromEnd'] = peak_bed_file['chromEnd'].values.astype(np.int32)
    
    
def read_motif_file(motif_file,genes):
    motif_dir=os.getcwd()+'/sckinetics/motif_data/'
    motifs = read_motifs(motif_dir+motif_file)
    motifs_keep = []
    # to do: parallelize this
    for motif in motifs:
        motif_names = motif.factors['direct'] + motif.factors['indirect\nor predicted']
        motif_names=[motif.upper() for motif in motif_names]
        included_genes = np.intersect1d(motif_names,genes)
        if len(included_genes)>0:
            motif.factors['included'] = included_genes
            motifs_keep.append(motif)
    return motifs_keep

def run_MAGIC(adata, knn = 30, t = 3, use_rep = 'X_pca', neighbors_key = 'neighbors', n_components = 30, 
              metric = 'euclidean'):
    N = adata.shape[0]
    ka_val = int(np.ceil(knn/3))
    nbrs = NearestNeighbors(n_neighbors=int(knn), metric=metric).fit(adata.obsm[use_rep])
    sparse_distance_matrix = nbrs.kneighbors_graph(adata.obsm[use_rep], mode='distance')
    row, col, val = sparse.find(sparse_distance_matrix)
    ka_list = np.asarray([np.sort(val[row == j])[ka_val] for j in np.unique(row)])
    scaled_distance_matrix = sparse.csr_matrix(sparse_distance_matrix/ka_list[:, None])
    x, y, scaled_dists = sparse.find(scaled_distance_matrix)
    W = sparse.csr_matrix((np.exp(-scaled_dists), (x, y)), shape=[N, N])
    W.setdiag(1)
    kernel = W + W.T
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = sparse.csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
    D, V = sparse.linalg.eigs(T, n_components, tol=1e-4, maxiter=1000)
    D, V = np.real(D), np.real(V)
    inds = np.argsort(D)[::-1]
    D, V = D[inds], V[:, inds]
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])
    imputed_data_temp = adata.X.copy()
    for steps in range(t):
        imputed_data_temp = T * imputed_data_temp
    adata.obsm['X_magic'] = imputed_data_temp.toarray()
    
    return adata

# """
# : Progress Bar
# """
# @contextlib.contextmanager
# def tqdm_joblib(tqdm_object):

#     def tqdm_print_progress(self):
#         if self.n_completed_tasks > tqdm_object.n:
#             n_completed = self.n_completed_tasks - tqdm_object.n
#             tqdm_object.update(n=n_completed)

#     original_print_progress = joblib.parallel.Parallel.print_progress
#     joblib.parallel.Parallel.print_progress = tqdm_print_progress

#     try:
#         yield tqdm_object
#     finally:
#         joblib.parallel.Parallel.print_progress = original_print_progress
#         tqdm_object.close()
