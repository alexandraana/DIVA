#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import random
import torch.autograd.profiler as profiler
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D

writer = SummaryWriter()

import scIB
import scanpy as sc

DATA_PATH = '/ngc/people/aleana/simulations/scibsim1.txt'


expr = pd.read_csv(DATA_PATH, sep=',')
exprdata = expr.drop(columns=['x', 'Batch', 'Cell', 'Group'])
exprdata.to_csv('exprdata.csv')


exprdata = sc.read_csv('exprdata.csv')
exprdata.obs['Batch'] = expr['Batch'].to_list()
exprdata.obs['Group'] = expr['Group'].to_list()


sc.pp.neighbors(exprdata, use_rep='X')


sc.tl.umap(exprdata)
sc.pl.umap(exprdata, color=['Batch', 'Group'])


from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(res):
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Batch1'], ['Batch2'], ['Batch3'], ['Batch4'], ['Batch5'], ['Batch6']]
    enc.fit(X)
    Y = [[res]]
    array = enc.transform(Y).toarray()

    return (array)

from sklearn.preprocessing import OneHotEncoder


def one_hot_encodey(res):
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Group1'], ['Group2'], ['Group3'], ['Group4'], ['Group5'], ['Group6'], ['Group7']]
    enc.fit(X)
    Y = [[res]]
    array = enc.transform(Y).toarray()
    return (array)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device



def load_data(path):
    # read in from csv
    expressions = pd.read_csv(DATA_PATH, sep=',')
    display(expressions)
    subtypes = expressions['Batch']
    batches = subtypes
    subtypes
    celltype = expressions['Group']
    celltype
    lista = []
    lista2 = []
    expressions = expressions.drop(columns=['x', 'Batch', 'Cell', 'Group']).astype('float32')

    index = list(subtypes.index[subtypes == 'Batch1']) + list(subtypes.index[subtypes == 'Batch2']) + list(
        subtypes.index[subtypes == 'Batch3']) + list(subtypes.index[subtypes == 'Batch4']) + list(
        subtypes.index[subtypes == 'Batch5']) + list(subtypes.index[subtypes == 'Batch6'])

    # random.shuffle(index)
    expressions = expressions.iloc[index, :].astype('float32')
    print(type(expressions))
    subtypes = subtypes.iloc[index]
    batches = batches.iloc[index]
    celltype = celltype.iloc[index]
    standardizer = preprocessing.StandardScaler()
    expressions = standardizer.fit_transform(expressions)

    subtypes = subtypes.tolist()
    for i in range(len(subtypes)):
        lista.append(one_hot_encode(subtypes[i]))

    celltype = celltype.tolist()
    for i in range(len(celltype)):
        lista2.append(one_hot_encodey(celltype[i]))
    print(expressions.shape)
    return expressions, standardizer, lista, lista2, subtypes, celltype


def numpyToTensor(expressions):
    x_train = torch.from_numpy(expressions).to(device)
    return x_train


from torch.utils.data import Dataset, DataLoader


class DataBuilder(Dataset):
    def __init__(self, path):
        self.expressions, self.standardizer, self.subtypes, self.celltype, self.batches_text, self.celltype_text = load_data(
            DATA_PATH)
        self.expressions = numpyToTensor(self.expressions)
        self.len = self.expressions.shape[0]

    def __getitem__(self, item):
        return self.expressions[item], self.subtypes[item], self.batches_text[item], self.celltype[item], \
               self.celltype_text[item]

    def __len__(self):
        return self.len


data_set = DataBuilder(DATA_PATH)
trainloader = DataLoader(dataset=data_set, batch_size=12500)


class px(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(px, self).__init__()

        #         # Sampling vector
        print(zx_dim + zd_dim + zy_dim)
        self.fc3 = nn.Linear(zx_dim + zd_dim + zy_dim, zx_dim + zd_dim + zy_dim)
        self.fc_bn3 = nn.BatchNorm1d(zx_dim + zd_dim + zy_dim)
        self.fc4 = nn.Linear(zx_dim + zd_dim + zy_dim, hidden1)
        self.fc_bn4 = nn.BatchNorm1d(hidden1)

        #         # Decoder
        self.linear4 = nn.Linear(hidden1, hidden1)
        self.lin_bn4 = nn.BatchNorm1d(num_features=hidden1)
        self.linear5 = nn.Linear(hidden1, hidden2)
        self.lin_bn5 = nn.BatchNorm1d(num_features=hidden2)
        self.linear6 = nn.Linear(hidden2, x_dim)
        self.lin_bn6 = nn.BatchNorm1d(num_features=x_dim)

        self.relu = nn.ReLU()

    def forward(self, zx, zd, zy):
        z = torch.cat((zx, zd), 1)
        z = torch.cat((z, zy), 1)
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))


class pzd(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(pzd, self).__init__()

        self.linear1 = nn.Linear(d_dim, hidden1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden2, zd_dim)
        self.bn1 = nn.BatchNorm1d(num_features=zd_dim)
        self.fc21 = nn.Linear(zd_dim, zd_dim)
        self.fc22 = nn.Linear(zd_dim, zd_dim)

        self.relu = nn.ReLU()

    def forward(self, d):
        lin1 = self.relu(self.lin_bn1(self.linear1(d)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2


class pzy(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(pzy, self).__init__()

        self.linear1 = nn.Linear(y_dim, hidden1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden2, zy_dim)
        self.bn1 = nn.BatchNorm1d(num_features=zy_dim)
        self.fc21 = nn.Linear(zy_dim, zy_dim)
        self.fc22 = nn.Linear(zy_dim, zy_dim)

        self.relu = nn.ReLU()

    def forward(self, y):
        lin1 = self.relu(self.lin_bn1(self.linear1(y)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

class qzd(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(qzd, self).__init__()

        self.linear1 = nn.Linear(x_dim, hidden2)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden2)
        self.linear2 = nn.Linear(hidden2, hidden1)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden1)
        self.linear3 = nn.Linear(hidden1, hidden1)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden1)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden1, zd_dim)
        self.bn1 = nn.BatchNorm1d(num_features=zd_dim)
        self.fc21 = nn.Linear(zd_dim, zd_dim)
        self.fc22 = nn.Linear(zd_dim, zd_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2


class qzy(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(qzy, self).__init__()

        self.linear1 = nn.Linear(x_dim, hidden2)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden2)
        self.linear2 = nn.Linear(hidden2, hidden1)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden1)
        self.linear3 = nn.Linear(hidden1, hidden1)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden1)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden1, zy_dim)
        self.bn1 = nn.BatchNorm1d(num_features=zy_dim)
        self.fc21 = nn.Linear(zy_dim, zy_dim)
        self.fc22 = nn.Linear(zy_dim, zy_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

class qzx(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(qzx, self).__init__()

        self.linear1 = nn.Linear(x_dim, hidden2)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden2)
        self.linear2 = nn.Linear(hidden2, hidden1)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden1)
        self.linear3 = nn.Linear(hidden1, hidden1)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden1)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden1, zx_dim)
        self.bn1 = nn.BatchNorm1d(num_features=zx_dim)
        self.fc21 = nn.Linear(zx_dim, zx_dim)
        self.fc22 = nn.Linear(zx_dim, zx_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

class qd(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden2, hidden1, zx_dim, zd_dim, zy_dim):
        super(qd, self).__init__()

        self.linear1 = nn.Linear(zd_dim, hidden1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden2, d_dim)
        self.bn1 = nn.BatchNorm1d(num_features=d_dim)

        self.relu = nn.ReLU()

    def forward(self, zd):
        lin1 = self.relu(self.lin_bn1(self.linear1(zd)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        r = F.relu(self.bn1(self.fc1(lin3)))

        return r

class qy(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden2, hidden1, zx_dim, zd_dim, zy_dim):
        super(qy, self).__init__()

        self.linear1 = nn.Linear(zy_dim, hidden1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden2)
        self.linear3 = nn.Linear(hidden2, hidden2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=hidden2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden2, y_dim)
        self.bn1 = nn.BatchNorm1d(num_features=y_dim)

        self.relu = nn.ReLU()

    def forward(self, zy):
        lin1 = self.relu(self.lin_bn1(self.linear1(zy)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        r = F.relu(self.bn1(self.fc1(lin3)))

        return r

class DIVA(nn.Module):
    def __init__(self, x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim):
        super(DIVA, self).__init__()
        hidden1x = 800
        hidden2x = 5000
        hidden1d = 5
        hidden2d = 8
        hidden1y = 20
        hidden2y = 38
        zx_dim = 5
        zd_dim = 10
        zy_dim = 50

        self.px = px(x_dim, d_dim, y_dim, hidden1x, hidden2x, zx_dim, zd_dim, zy_dim)
        self.pzd = pzd(x_dim, d_dim, y_dim, hidden1d, hidden2d, zx_dim, zd_dim, zy_dim)
        self.pzy = pzy(x_dim, y_dim, y_dim, hidden1y, hidden2y, zx_dim, zd_dim, zy_dim)

        self.qzd = qzd(x_dim, d_dim, y_dim, hidden1x, hidden2x, zx_dim, zd_dim, zy_dim)
        self.qzx = qzx(x_dim, d_dim, y_dim, hidden1x, hidden2x, zx_dim, zd_dim, zy_dim)
        self.qzy = qzy(x_dim, d_dim, y_dim, hidden1x, hidden2x, zx_dim, zd_dim, zy_dim)

        self.qd = qd(x_dim, d_dim, y_dim, hidden1d, hidden2d, zx_dim, zd_dim, zy_dim)
        self.qy = qy(x_dim, d_dim, y_dim, hidden1y, hidden2y, zx_dim, zd_dim, zy_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, d, y):

        mud, logvard = self.qzd(x)
        zd = self.reparameterize(mud, logvard)

        muy, logvary = self.qzy(x)
        zy = self.reparameterize(muy, logvary)

        mux, logvarx = self.qzx(x)
        zx = self.reparameterize(mux, logvarx)

        x_recon = self.px(zx, zd, zy)

        mu_dp, logvar_dp = self.pzd(d)
        zd_p = self.reparameterize(mu_dp, logvar_dp)

        d_recon = self.qd(zd)

        mu_yp, logvar_yp = self.pzy(y)
        zy_p = self.reparameterize(mu_yp, logvar_yp)

        y_recon = self.qy(zy)

        return x_recon, mux, logvarx, mud, muy, mu_dp, logvar_dp, d_recon, mu_yp, logvar_yp, y_recon


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mux, logvarx, d_recon, d, mud, logvard, y_recon, y, muy, logvary):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_MSEd = self.mse_loss(d_recon, d)
        loss_MSEy = self.mse_loss(y_recon, y)
        loss_KLDx = -0.5 * torch.sum(1 + logvarx - mux.pow(2) - logvarx.exp())
        loss_KLDd = -0.5 * torch.sum(1 + logvard - mud.pow(2) - logvard.exp())
        loss_KLDy = -0.5 * torch.sum(1 + logvary - muy.pow(2) - logvary.exp())

        alphad = 5000
        alphay = 5000

        return loss_MSE + loss_KLDx + loss_KLDd + loss_MSEd * alphad + loss_KLDy + loss_MSEy * alphay, loss_MSE + loss_KLDx, loss_KLDd + loss_MSEd * alphad, loss_KLDy + loss_MSEy * alphay


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

x_dim = data_set.expressions.shape[1]
d_dim = len(data_set.subtypes[0][0])
y_dim = len(data_set.celltype[0][0])

hidden1 = 100
hidden2 = 100
zx_dim = 100
zd_dim = 100
zy_dim = 100
model = DIVA(x_dim, d_dim, y_dim, hidden1, hidden2, zx_dim, zd_dim, zy_dim).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_mse = customLoss()

epochs = 7000
log_interval = 100
val_losses = []
train_losses = []

def train(epoch):
    model.train()
    train_loss = 0
    tlx = 0
    tld = 0
    tly = 0
    trainloader
    for batch_idx, (x, d, b, y, c) in enumerate(trainloader):

        d = d.to(device)
        d = d.squeeze(1)
        d = d.float()

        y = y.to(device)
        y = y.squeeze(1)
        y = y.float()

        optimizer.zero_grad()
        recon_batch, mu, logvar, muzd_byx, muyx, mud, logvard, d_recon, mu_yp, logvar_yp, y_recon = model(x, d, y)

        loss, lx, ld, ly = loss_mse(recon_batch, x, mu, logvar, d_recon, d, mud, logvard, y_recon, y, mu_yp, logvar_yp)
        if (loss < 0):
            print(loss)
        loss.backward()
        train_loss += loss.item()
        tlx += lx.item()
        tld += ld.item()
        tly += ly.item()
        optimizer.step()

    if epoch % 100 == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(trainloader.dataset)))
        writer.add_scalar("LOSS", train_loss / len(trainloader.dataset), epoch)
        train_losses.append(train_loss / len(trainloader.dataset))
        with torch.no_grad():
            print('loss x: ', tlx / len(trainloader.dataset))
            print('loss d: ', tld / len(trainloader.dataset))
            print('loss y: ', tly / len(trainloader.dataset))

            writer.add_scalar("Zx loss", tlx / len(trainloader.dataset), epoch)
            writer.add_scalar("Zd loss", tld / len(trainloader.dataset), epoch)
            writer.add_scalar("Zy loss", tly / len(trainloader.dataset), epoch)
            writer.add_scalars("Zx, zd, zy  losses",
                               {"zx loss": tlx / len(trainloader.dataset), "zd loss": tld / len(trainloader.dataset),
                                "zy loss": tly / len(trainloader.dataset)}, epoch)

        writer.add_histogram('linear1pzd.weight', model.pzd.linear1.weight, epoch)
        writer.add_histogram('linear1pzy.weight', model.pzy.linear1.weight, epoch)
        writer.add_histogram('linear1qd.weight', model.qd.linear1.weight, epoch)
        writer.add_histogram('linear1qy.weight', model.qy.linear1.weight, epoch)
        writer.add_histogram('linear1qzx.weight', model.qzx.linear1.weight, epoch)
        writer.add_histogram('linear1qzy.weight', model.qzy.linear1.weight, epoch)
        writer.add_histogram('linear1qzd.weight', model.qzd.linear1.weight, epoch)
        writer.add_histogram('fc3px.weight', model.px.fc3.weight, epoch)

        writer.add_histogram('fc1qzd.weight', model.qzd.fc1.weight, epoch)
        writer.add_histogram('fc21qzd.weight', model.qzd.fc21.weight, epoch)
        writer.add_histogram('fc22qzd.weight', model.qzd.fc22.weight, epoch)

        writer.add_histogram('fc1qzy.weight', model.qzy.fc1.weight, epoch)
        writer.add_histogram('fc21qzy.weight', model.qzy.fc21.weight, epoch)
        writer.add_histogram('fc22qzy.weight', model.qzy.fc22.weight, epoch)

        writer.add_histogram('fc1qzx.weight', model.qzx.fc1.weight, epoch)
        writer.add_histogram('fc21qzx.weight', model.qzx.fc21.weight, epoch)
        writer.add_histogram('fc22qzx.weight', model.qzx.fc22.weight, epoch)

        writer.add_histogram('fc1pzd.weight', model.pzd.fc1.weight, epoch)
        writer.add_histogram('fc21pzd.weight', model.pzd.fc21.weight, epoch)
        writer.add_histogram('fc22pzd.weight', model.pzd.fc22.weight, epoch)

        writer.add_histogram('fc1pzy.weight', model.pzy.fc1.weight, epoch)
        writer.add_histogram('fc21pzy.weight', model.pzy.fc21.weight, epoch)
        writer.add_histogram('fc22pzy.weight', model.pzy.fc22.weight, epoch)

        writer.add_histogram('linear2pzd.weight', model.pzd.linear2.weight, epoch)
        writer.add_histogram('linear2qd.weight', model.qd.linear2.weight, epoch)
        writer.add_histogram('linear2pzy.weight', model.pzy.linear2.weight, epoch)
        writer.add_histogram('linear2qy.weight', model.qy.linear2.weight, epoch)
        writer.add_histogram('linear2qzx.weight', model.qzx.linear2.weight, epoch)
        writer.add_histogram('linear2qzd.weight', model.qzd.linear2.weight, epoch)
        writer.add_histogram('linear2qzy.weight', model.qzy.linear2.weight, epoch)
        writer.add_histogram('fc4px.weight', model.px.fc4.weight, epoch)

        writer.add_histogram('linear3pzd.weight', model.pzd.linear3.weight, epoch)
        writer.add_histogram('linear3qd.weight', model.qd.linear3.weight, epoch)
        writer.add_histogram('linear3pzy.weight', model.pzy.linear3.weight, epoch)
        writer.add_histogram('linear3qy.weight', model.qy.linear3.weight, epoch)
        writer.add_histogram('linear3qzx.weight', model.qzx.linear3.weight, epoch)
        writer.add_histogram('linear3qzd.weight', model.qzd.linear3.weight, epoch)
        writer.add_histogram('linear3qzy.weight', model.qzy.linear3.weight, epoch)
        writer.add_histogram('linear4px.weight', model.px.linear4.weight, epoch)
        writer.add_histogram('linear5px.weight', model.px.linear5.weight, epoch)
        writer.add_histogram('linear6px.weight', model.px.linear6.weight, epoch)

    return mu, mud, recon_batch, muzd_byx, muyx, x, d, b, y, c


for epoch in range(1, epochs + 1):
    mu, mud_output, save_recon, mu_dbyx_output, muyx, x_pca, data, batches, celltype, celltypes_text = train(epoch)

writer.flush()
writer.close()

pd.DataFrame(mu.cpu().detach().numpy()).to_csv('zxdata.csv', header=False, index=False)
pd.DataFrame(muyx.cpu().detach().numpy()).to_csv('zydata.csv', header=False, index=False)
pd.DataFrame(mu_dbyx_output.cpu().detach().numpy()).to_csv('zddata.csv', header=False, index=False)

zxdata = sc.read_csv('zxdata.csv')
zydata = sc.read_csv('zydata.csv')
zddata = sc.read_csv('zddata.csv')


zxdata.obs['Batch'] = batches
zxdata.obs['Group'] = celltypes_text
zydata.obs['Batch'] = batches
zydata.obs['Group'] = celltypes_text
zddata.obs['Batch'] = batches
zddata.obs['Group'] = celltypes_text


sc.pp.pca(exprdata, svd_solver='arpack')

sc.pp.neighbors(zxdata)
sc.pp.neighbors(zydata)
sc.pp.neighbors(zddata)


sc.tl.umap(zxdata)
sc.pl.umap(zxdata, color=['Batch'])
sc.pl.umap(zxdata, color=['Group'])

sc.tl.umap(zddata)
sc.pl.umap(zddata, color=['Batch'])
sc.pl.umap(zddata, color=['Group'])

sc.tl.umap(zydata)
sc.pl.umap(zydata, color=['Batch'])
sc.pl.umap(zydata, color=['Group'])

sc.pp.pca(zxdata, svd_solver='arpack')
sc.pp.pca(zydata, svd_solver='arpack')

scIB.me.pc_regression(exprdata.X, exprdata.obs["Batch"])

score = scIB.me.pcr_comparison(
    exprdata, zxdata,
    covariate='Batch',
    n_comps=50,
    scale=True
)

score = scIB.me.pcr_comparison(
    exprdata, zydata,
    covariate='Batch',
    scale=True
)

score = scIB.me.silhouette(
    zydata,
    group_key='Group',
    embed='X_pca',
    scale=True
)

zxdata.obsp['distances'] = zxdata.uns['neighbors']['distances']
zxdata.obsp['connectivities'] = zxdata.uns['neighbors']['connectivities']
zydata.obsp['distances'] = zydata.uns['neighbors']['distances']
zydata.obsp['connectivities'] = zydata.uns['neighbors']['connectivities']

scIB.me.lisi_graph(zydata, batch_key='Batch', label_key='Group')


scIB.me.lisi_graph(zydata, batch_key='Batch', label_key='Group')

scIB.me.kBET(zydata, batch_key='Batch', label_key='Group')

scIB.me.graph_connectivity(zydata, label_key='Group')

_, sil = scIB.me.silhouette_batch(
    zydata,
    batch_key='Batch',
    group_key='Group',
    embed='X_pca',
    scale=True,
    verbose=False
)
score = sil['silhouette_score'].mean()

scIB.me.isolated_labels(zydata, batch_key='Batch', label_key='Group', embed='X_pca', cluster=True, verbose=False)

scIB.me.isolated_labels(zydata, batch_key='Batch', label_key='Group', embed='X_pca', cluster=False, verbose=False)


