#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np 
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def create_file_x_y_t(stack: np.ndarray,
                        list_timesteps:list[str]
                        ):
    """
    Creates a temporal edge list format from a dynamic network represented as a stack of adjacency matrices.
    
    Input : 
        - stack (np.ndarray) : A 3D numpy array of shape (T, N, N) where:
            - T is the number of time steps
            - N is the number of nodes
            Each slice stack[t,:,:] represents the adjacency matrix at time t
        
        - list_timesteps (list[str]) : a list of timestamps as strings, corresponding to each layer in the stack
        
    Returns : 
        - txt (str) : A string containing the temporal edge list format where each line follows:
        "source_node target_node normalized_time"
            - source_node and target_node are indices of the nodes
            - normalized_time is scaled between 0 and 1
    
    Notes :
    - Only upper triangular edges are kept (directed edges in one direction)
    - Timestamps are normalized to [0,1] range
    - Edge format: "i j t" where i,j are node indices and t is normalized time
    """
    
    serie_timestamp=pd.to_datetime(pd.Series(list_timesteps)).astype("int64")
    normalized_tstamp=(serie_timestamp - serie_timestamp.min())/(serie_timestamp.max()-serie_timestamp.min())
    
    txt=""
    coord_triu=np.triu_indices(stack[0,:,:].shape[0],k=1)
    coord_triu=[(coord_triu[0][i],coord_triu[1][i]) for i in range(coord_triu[0].shape[0])]

    for idx,val in normalized_tstamp.items() : 
        coordonnees = np.argwhere(stack[idx,:,:]==1)
        
        for couple in coordonnees: 
            if tuple(couple) in coord_triu: ## Pour n'avoir qu'une seule arrête, à enlever si on veut les arrêtes dans les deux sens
            
                txt+="{} {} {} \n".format(*couple,val)
    return txt



def create_file_nodeId_label(list_id : list[str]) : 
    """
    Creates a node label file where each line contains a node ID and its label.
    
    Input:
        - list_id (list[str]): A list of node identifiers
        
    Returns:
        - txt (str): A string containing node labels in the format:
        "node_id label"
            - node_id is the index of the node (0 to len(list_id)-1)
            - label is set to 1 for all nodes (default label)
    
    Notes:
        - All nodes are assigned the same label (1)
        - Node indices are generated sequentially starting from 0
        - Each line follows the format: "i 1" where i is the node index
    """
    
    txt = "" 
    for i in range(len(list_id)) : 
        txt += "{} 1 \n".format(i) ### Nos noeuds ne sont pas annotés, on met l'étiquette 1 parotut 
    
    return txt



def convert_HTNE_embFile(path : str,
                        max_row : int,
                        skiprows : int = 1 
                        ) -> np.ndarray:
    """
    Loads and converts HTNE (Hierarchical Temporal Network Embedding) embedding file to numpy array.
    
    Input:
        - path (str): Path to the HTNE embedding file
        - max_row (int): Maximum number of rows to read from the file
        - skiprows (int, default=1): Number of rows to skip at the beginning of the file
        
    Returns:
        - np.ndarray: A numpy array containing the node embeddings where:
            - Each row represents a node
            - Each column represents a dimension of the embedding
    """
    return np.loadtxt(path,max_rows=max_row, skiprows=skiprows)


def TSNE_show(matrix : np.matrix) : 
    
    output = TSNE().fit_transform(matrix)
    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(output[:, 0], output[:, 1], cmap="jet", alpha=0.7)
    
    plt.colorbar(scatter, label="Classes")
    plt.title("Visualisation t-SNE des données")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
    

###############################################################################
###############################################################################
###############################################################################
############################################################################### 
###############################################################################

import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import sys

from torch.utils.data import Dataset

FType = torch.FloatTensor
LType = torch.LongTensor
if torch.backends.mps.is_available(): 
    DID = torch.device("mps")
    

class HTNE_a:
    def __init__(self, file_path, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=15):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = HTNEDataSet(file_path, neg_size, hist_len, directed)
        self.node_dim = self.data.get_node_dim()

        if torch.backends.mps.is_available():
            with torch.device("mps"):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).to("mps"), requires_grad=True)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).to("mps"), requires_grad=True)

                self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                    FType).to("mps")), requires_grad=True)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

            self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                FType)), requires_grad=True)

        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.att_param, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb)**2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb)**2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1))**2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)
        
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1))**2).sum(dim=3).neg()


        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (Variable(h_time_mask).unsqueeze(2))).sum(dim=1)
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.backends.mps.is_available():
            with torch.device("mps"):
                p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times,
                                                h_time_mask)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.backends.mps.is_available():
            with torch.device("mps"):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=5)
            # if epoch % self.save_step == 0 and epoch != 0:
            #     #torch.save(self, './model/dnrl-dblp-%d.bin' % epoch)
            #     self.save_node_embeddings('./emb/cows_htne_attn_%d.emb' % (epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    sys.stdout.flush()

                if torch.backends.mps.is_available():
                    with torch.device("mps"):
                        self.update(sample_batched['source_node'].type(LType).to("mps"),
                                    sample_batched['target_node'].type(LType).to("mps"),
                                    sample_batched['target_time'].type(FType).to("mps"),
                                    sample_batched['neg_nodes'].type(LType).to("mps"),
                                    sample_batched['history_nodes'].type(LType).to("mps"),
                                    sample_batched['history_times'].type(FType).to("mps"),
                                    sample_batched['history_masks'].type(FType).to("mps"))
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                                str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()
            self.save_node_embeddings('./emb/cows_htne_attn_%d.emb' % (epoch))


    def save_node_embeddings(self, path):
        
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        
        if torch.backends.mps.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()

class HTNEDataSet(Dataset):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform

        # self.max_d_time = -sys.maxint  # Time interval [0, T]
        self.max_d_time = -sys.maxsize
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                d_time = float(parts[2])  # time slot, delta t

                self.node_set.update([s_node, t_node])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        self.node_dim = len(self.node_set)

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        print(self.node_dim)
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]
        
        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling()
        
        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'neg_nodes': neg_nodes,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes


    
    
    

