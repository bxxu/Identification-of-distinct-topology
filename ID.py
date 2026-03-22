import numpy as np
import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import math

import scipy
import random
import statistics
import umap as UM
from scipy.stats import zscore
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import seaborn as sns
from IPython.display import clear_output


def train_model(data, Nnodes = 500, latent_dim = 10, fix = True, batchSize = 150, Ndir = 50, Niter = 30000):
    fix = fix
    
    VAE = simpleVAE(inputDim=data.shape[1], Nnodes = Nnodes, latent_dim =latent_dim, fixStd=fix, fixval=0.05)
    optimizer_VAE = torch.optim.Adam(VAE.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=0.0001) 

    record_pcf = np.array([])
    record_mov = np.array([])
    record_recon = np.array([])
    
    batchSize = batchSize
    Ndir = Ndir
    mvn = torch.distributions.MultivariateNormal(torch.zeros(VAE.latentDim), torch.eye(VAE.latentDim))
    
    for i in range(Niter):
        optimizer_VAE.zero_grad()


        random_sample = random.sample(range(data.shape[0]),batchSize)
        x = data[random_sample,:]  

        x_hat = VAE(x)
        loss_pcf = torch.tensor([0.0,0.0])
        loss_movement = torch.tensor([0.0,0.0])
        # recon_loss = torch.log(((x-x_hat)**2).sum(dim = 1)).mean()
        recon_loss = ((x-x_hat)**2).sum(dim = 1).mean()

        if fix == False:
            loss = ((x-x_hat)**2).sum(dim = 1).mean() +  1 * VAE.KL_loss.mean()
        if fix == True:
            W_dist = sample_wasserstein(mvn, VAE.lowdim, Ndir)
            loss = ((x-x_hat)**2).sum(dim = 1).mean() + 5000 * W_dist  #+ 300 * loss_movement.mean()#ori
            record_recon = np.append(record_recon, ((x-x_hat)**2).sum(dim = 1).mean().detach().numpy())
        loss.backward()
        optimizer_VAE.step()

        if np.mod(i, 1000) == 0:
            print('Iter: {}, loss_re: {:.4f}, loss_pcf: {:.4f}'
                  .format(i, loss, loss_pcf.abs().mean().detach().numpy()))    
            clear_output(wait=True)
            
    return VAE

    
    
    
def get_gene_embedding_ID(model, all_data_tensor, Ntrial = 50000, Nclus = 2):
    mvn = torch.distributions.MultivariateNormal(torch.zeros(model.latentDim), torch.eye(model.latentDim))

    all_hat = model(all_data_tensor)
    all_grad = np.zeros([Ntrial, all_data_tensor.shape[1]])
    loc = np.zeros([Ntrial, model.latentDim])
    DIR = np.zeros([Ntrial, model.latentDim])
    mag = np.array([])
    dir_norm = np.array([])
    cell_idx = np.array([])
    
    for i in range(Ntrial):
        ssample = random.sample(range(all_data_tensor.shape[0]),1)

        n = mvn.sample([1]) * 1 #* 0.0001 *
        dir1 = np.random.randint(low = 0, high = model.latentDim, size = 1)[0]
        n = n/torch.sqrt((n**2).sum()) * 0.1 #ori is 0.1, 
        # n[0,dir1] = 0.1
        z = model.lowdim[ssample, :].detach()
        out = model.back2high(z)
        out_hat = model.back2high(z + n)

        exp_change = out_hat - out
        # exp_change[exp_change.abs() < exp_change.abs().median()] = 0
        # scale_factor = 
        # exp_change = exp_change / exp_change.std()
        all_grad[i,:] = exp_change.detach().numpy() 
        loc[i,:] = z
        DIR[i,:] = n
        mag = np.append(mag, (exp_change**2).detach().numpy().sum())
        dir_norm = np.append(dir_norm, (n.detach().numpy()**2).sum())
        cell_idx = np.append(cell_idx, ssample)
        
    reducer_gene = UM.UMAP(n_neighbors= 5, 
                   metric = 'euclidean', 
                   repulsion_strength=1, 
                   n_epochs= 1500, 
                   negative_sample_rate=5, 
                   min_dist = .1, #ori was 0.5? but 0.1 works better for real scrna-seq data
                   verbose = False,
                   random_state=1,
                   n_components = 2
                  )

    pca_perturb = PCA(n_components=20, random_state = 1)  # smaller number of pcs give clearer cluster?, ori is 50
    raw_data = all_grad.copy()
    data =  np.abs(all_grad.T)
    data = zscore(data)
    store = data.copy()
    data = pca_perturb.fit_transform(data)

    embedding_grad = reducer_gene.fit_transform(data) #original did not use abs
    
    kmeans = KMeans(n_clusters=Nclus, random_state=0)
    spectral = SpectralClustering(n_clusters=Nclus,
            assign_labels='discretize',
            random_state=0)
    kmeans.fit(embedding_grad)
    # spectral.fit(embedding_grad)
    
    return mag, raw_data, embedding_grad, cell_idx

class latent_module_mean(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = self.fc4(z)
        
        return z

class latent_module_std(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        
        return z
    
class back2high(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self,z):
        ori_data = torch.relu(self.fc1(z))
        ori_data = torch.relu(self.fc2(ori_data))
        ori_data = torch.relu(self.fc3(ori_data))
        ori_data = self.fc4(ori_data)
        
        return ori_data
    
class simpleVAE(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim, fixStd = True, fixval = 0.5):
        super().__init__()
        
        self.latentMean = latent_module_mean(inputDim, Nnodes, latent_dim)
        self.latentStd = latent_module_std(inputDim, Nnodes, latent_dim)
        self.back2high = back2high(inputDim, Nnodes, latent_dim)
        
        self.latentDim = latent_dim
        self.fixStd = fixStd
        self.fixVal = fixval
        
        
        
    def forward(self,z):
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim), torch.eye(self.latentDim))
        
        latent_mean = self.latentMean(z)
        latentStd = torch.clamp(self.latentStd(z), min=1e-3)
        
        if self.fixStd == False:
            latent_state = latent_mean + mvn.sample([1]) * latentStd
        
        if self.fixStd == True:
            latent_state = latent_mean + mvn.sample([1]) * self.fixVal

        ori_data = self.back2high(latent_state)
        
        
        self.KL_loss = 0.5 *  ((latent_mean**2).sum(dim = 1)  - torch.clamp(torch.log(latentStd**2), min = -100).sum(dim = 1) - (self.latentDim - 1) + (latentStd**2).sum(dim = 1) )
        self.lowdim = latent_mean
        self.lowstd = latentStd
        return ori_data
    
    # def train_VAE(self, train_data, pcf_loss, batchSize = 150, Ndir = 50, Nitr = 20000, pcf_weight = 150):
    #     fix = self.fixStd
    #     pcf = pcf_loss
    #     record_pcf = np.array([])
    #     record_pcf_max = np.array([])
    #     optimizer_VAE = torch.optim.Adam(self.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=0.0001) 

    #     mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim), torch.eye(self.latentDim))

    #     for i in range(Nitr):
    #         optimizer_VAE.zero_grad()


    #         random_sample = random.sample(range(train_data.shape[0]),batchSize)
    #         x = train_data[random_sample,:]
    #         x_hat = self.forward(x)
    #         recon_loss = torch.log(((x-x_hat)**2).sum(dim = 1)).mean()
            
    #         loss_pcf = torch.tensor([0.0,0.0])


    #         if fix == False:
    #             loss = recon_loss +  self.latentDim * self.KL_loss.mean()
    #         if fix == True:
    #             if pcf == True:
    #                 if np.mod(i, 5) == 0:
    #                     loss_pcf = ortho_loss_numerical_v2(self,self.lowdim , eps = 0.1)
    #                     record_pcf = np.append(record_pcf, loss_pcf.abs().mean().detach().numpy())
    #                     record_pcf_max = np.append(record_pcf_max, loss_pcf.abs().max().detach().numpy())

    #             else:
    #                 if np.mod(i, 5) == 0:
    #                     loss_pcf_record = ortho_loss_numerical_v2(self,self.lowdim , eps = 0.1)
    #                     record_pcf = np.append(record_pcf, loss_pcf_record.abs().mean().detach().numpy())
    #                     record_pcf_max = np.append(record_pcf_max, loss_pcf_record.abs().max().detach().numpy())
                        
    #             W_dist = sample_wasserstein(mvn, self.lowdim, Ndir)
    #             # loss = recon_loss + 5000 * W_dist + pcf_weight * loss_pcf.abs().mean()
    #             loss = ((x-x_hat)**2).sum(dim = 1).mean() + 5000 * W_dist + pcf_weight * loss_pcf.abs().mean() #ori
                
    #         loss.backward()
    #         optimizer_VAE.step()

    #         if np.mod(i, 1000) == 0:
    #             print('Iter: {}, loss_re: {:.4f}'
    #                   .format(i, recon_loss, ))    
    #             # clear_output(wait=True)
                
                
        train_hat = self.forward(train_data)
        # pred_accuracy = np.corrcoef(train_data.flatten().detach().numpy(), train_hat.flatten().detach().numpy())[0,1]
        pred_accuracy = ((train_data-train_hat)**2).sum(dim = 1).mean()
        self.accuracy = pred_accuracy
        self.record_pcf = record_pcf
        self.record_pcf_max = record_pcf_max

        return self
    


def sample_wasserstein(distribution, sample, dirNum):
    Ndim_samp = sample.shape[1]
    Nsamp = sample.shape[0]
    
    sampled_directions = torch.randn(dirNum, Ndim_samp)
    sampled_directions = sampled_directions / torch.linalg.norm(sampled_directions, dim = 1).view(sampled_directions.shape[0],1)
    dist_samples = distribution.sample([Nsamp])
    
    dist_projected = torch.matmul(dist_samples, sampled_directions.t())
    test_projected = torch.matmul(sample, sampled_directions.t())

    dist_projected_sort = torch.sort(dist_projected, dim = 0).values
    test_projected_sort = torch.sort(test_projected, dim = 0).values

    #return ((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).mean()
    #print(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).shape)
    return torch.sqrt(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1)).mean() / dirNum