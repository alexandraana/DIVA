import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import scanpy.external as sce
import bbknn
import scvi

DATA_PATH = '/ngc/people/aleana/simulations/scibsim1.txt'
expressions = pd.read_csv(DATA_PATH, sep=',')

adata = expressions.drop(columns=['x','Batch', 'Cell','Group'])

adata.to_csv('adata.csv')

andata = sc.read_csv('adata.csv')

andata.obs['Batch'] = expressions['Batch'].to_list()
andata.obs['Group'] = expressions['Group'].to_list()

sc.pp.neighbors(andata)
sc.tl.umap(andata)
sc.pl.umap(andata, color=['Batch','Group'])


from scvi.models import VAE
from sklearn.preprocessing import LabelEncoder
from scvi.inference import UnsupervisedTrainer
from scvi.dataset import AnnDatasetFromAnnData
n_epochs_scVI =np.min([round((20000/andata.n_obs)*400), 400])
n_epochs_scANVI = int(np.min([10, np.max([2, round(n_epochs_scVI / 3.)])]))
n_latent=5
n_hidden=128
n_layers=2

le = LabelEncoder()
net_adata = andata.copy()
net_adata.obs['batch_indices'] = le.fit_transform(net_adata.obs['Batch'].values)
net_adata.obs['labels'] = le.fit_transform(net_adata.obs['Group'].values)

net_adata = AnnDatasetFromAnnData(net_adata)

vae = VAE(
    net_adata.nb_genes,
    reconstruction_loss='nb',
    n_batch=net_adata.n_batches,
    n_layers=n_layers,
    n_latent=n_latent,
    n_hidden=n_hidden,
)

trainer = UnsupervisedTrainer(
    vae,
    net_adata,
    train_size=1.0,
    use_cuda=False,
)

trainer.train(n_epochs=n_epochs_scVI, lr=1e-3)

from scvi.inference import SemiSupervisedTrainer
from scvi.models import SCANVI
scanvi = SCANVI(net_adata.nb_genes, net_adata.n_batches, net_adata.n_labels,
                      n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, dispersion='gene', reconstruction_loss='nb')
scanvi.load_state_dict(trainer.model.state_dict(), strict=False)

# use default parameter from semi-supervised trainer class
trainer_scanvi = SemiSupervisedTrainer(scanvi, net_adata)
# use all cells as labelled set
trainer_scanvi.labelled_set = trainer_scanvi.create_posterior(trainer_scanvi.model, net_adata, indices=np.arange(len(net_adata)))
# put one cell in the unlabelled set
trainer_scanvi.unlabelled_set = trainer_scanvi.create_posterior(indices=[0])
trainer_scanvi.train(n_epochs=n_epochs_scANVI)

scanvi_full = trainer_scanvi.create_posterior(trainer_scanvi.model, net_adata, indices=np.arange(len(net_adata)))
latent, _, _ = scanvi_full.sequential().get_latent()

andata.obsm['X_emb'] = latent

import anndata
andata.obsm['X_emb'].shape
new = anndata.AnnData(andata.obsm['X_emb'])

new.obs['Batch'] = expressions['Batch'].to_list()
new.obs['Group'] = expressions['Group'].to_list()

sc.pp.neighbors(new)
sc.tl.umap(new)

sc.pl.umap(new, color=['Batch'])
sc.pl.umap(new, color=['Group'])

sc.pp.pca(new, svd_solver='arpack')

import scIB
score = scIB.me.pcr_comparison(
    andata, new,
    covariate='Batch',
    n_comps=50,
    scale=True
)

score = scIB.me.silhouette(
    new,
    group_key='Group',
    embed='X_pca',
    scale=True
)

new.obsp['distances'] = new.uns['neighbors']['distances']
new.obsp['connectivities'] = new.uns['neighbors']['connectivities']

score = scIB.me.lisi_graph(new,batch_key='Batch',label_key='Group')

scIB.me.lisi(new, batch_key='Batch', label_key='Group')

score = scIB.me.kBET(new,batch_key='Batch',label_key='Group')

score = scIB.me.graph_connectivity(new,label_key='Group')

_, sil = scIB.me.silhouette_batch(
    new,
    batch_key='Batch',
    group_key='Group',
    embed='X_pca',
    scale=True,
    verbose=False
)
score = sil['silhouette_score'].mean()

score = scIB.me.isolated_labels(new,batch_key='Batch',label_key='Group',embed='X_pca',cluster=True,verbose=True,n=4)

score = scIB.me.isolated_labels(
    new,
    label_key='Group',
    batch_key='Batch',
    embed='X_pca',
    n=4,
    cluster=False,
    verbose=True)