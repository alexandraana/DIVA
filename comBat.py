import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce
import bbknn
import scIB

DATA_PATH = '/ngc/people/aleana/simulations/scibsim1.txt'
expressions = pd.read_csv(DATA_PATH, sep=',')

adata = expressions.drop(columns=['x','Batch', 'Cell','Group'])

adata.to_csv('adata.csv')

andata = sc.read_csv('adata.csv')

andata.obs['Batch'] = expressions['Batch'].to_list()
andata.obs['Group'] = expressions['Group'].to_list()

adata = andata.copy()

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pp.pca(adata, svd_solver='arpack')

sc.pp.neighbors(andata)

sc.tl.umap(andata)
sc.pl.umap(andata, color=['Batch'])
sc.pl.umap(andata, color=['Group'])

sc.pp.pca(andata, svd_solver='arpack')

sc.pp.combat(andata,key='Batch')

sc.pp.pca(andata, svd_solver='arpack')

sc.pp.neighbors(andata)

sc.tl.umap(andata)

sc.pl.umap(andata, color=['Batch'])
sc.pl.umap(andata, color=['Group'])

score = scIB.me.pcr_comparison(
    adata, andata,
    covariate='Batch',
    n_comps=50,
    scale=True
)

score = scIB.me.silhouette(
    andata,
    group_key='Group',
    embed='X_pca',
    scale=True
)

andata.obsp['distances'] = andata.uns['neighbors']['distances']
andata.obsp['connectivities'] = andata.uns['neighbors']['connectivities']

score = scIB.me.lisi_graph(andata,batch_key='Batch',label_key='Group')

score = scIB.me.kBET(andata,batch_key='Batch',label_key='Group')

score = scIB.me.graph_connectivity(andata,label_key='Group')

_, sil = scIB.me.silhouette_batch(
    andata,
    batch_key='Batch',
    group_key='Group',
    embed='X_pca',
    scale=True,
    verbose=False
)
score = sil['silhouette_score'].mean()

scIB.me.isolated_labels(andata,batch_key='Batch',label_key='Group',embed='X_pca',cluster=True,verbose=False)

score = scIB.me.isolated_labels(
    andata,
    label_key='Group',
    batch_key='Batch',
    embed='X_pca',
    cluster=False,
    verbose=False
)