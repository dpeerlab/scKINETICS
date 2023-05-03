# scKINETICS
We introduce scKINETICS (Key regulatory Interaction NETwork for Inferring Cell Speed), a dynamical model of gene expression change which is fit with the simultaneous learning of per-cell transcriptional velocities and a governing gene regulatory network. This is accomplished through an expectation-maximization approach derived to learn the impact of each regulator on its target genes, leveraging biologically-motivated priors from epigenetic data, gene-gene co-expression, and constraints on cells’ future states imposed by the phenotypic manifold.

## Demo

The demo shows the main functionalities of the package including input file generation, GRN construction, EM estimation, visualizations, and the TF ablation experiment. 

## Data availability

Though the pancreas regeneration is not yet public, this method can work on any scRNAseq + bulk ATAC/scATAC/ChipSEQ/Multiome data.

## Readthedocs

[readthedocs](https://sckinetics.readthedocs.io/en/latest/index.html)
