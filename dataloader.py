import torch
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import json
import networkx as nx
import os
import sys
from torch.utils.data import DataLoader,TensorDataset
from itertools import cycle, chain

class Twibot22DomainData(InMemoryDataset):
    def __init__(self,graph_idx, root='data/',device='cpu',transform=None,pre_transform=None,pre_filter=None):
        self.root = root+'/'+str(graph_idx)
        self.graph_idx = graph_idx
        self.device = device
        super(Twibot22DomainData, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [f"/raw/group{self.graph_idx}_raw_feat.csv", f"/raw/group{self.graph_idx}_edges_new.csv",f"/raw/group{self.graph_idx}_nodes.csv",f"/raw/group{self.graph_idx}_graphlet.npy"]

    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def process(self):
        self.feats  = pd.read_csv(self.root+'/'+self.raw_file_names[0])
        self.edges  = pd.read_csv(self.root+'/'+self.raw_file_names[1])
        self.labels = pd.read_csv(self.root+'/'+self.raw_file_names[2])
        # user -> id
        self.id_map = {}
        id_ = 0
        for user in self.labels['id'].tolist():
            self.id_map[user] = id_
            id_ += 1
        # id -> user
        self.id_map_r = {id:user for user,id in self.id_map.items()}
        # replace
        self.feats['id']        = self.feats['id'].map(lambda x:self.id_map[x])
        self.labels['id']       = self.labels['id'].map(lambda x:self.id_map[x])
        self.edges['source']    = self.edges['source'].map(lambda x:self.id_map[x])
        self.edges['target']    = self.edges['target'].map(lambda x:self.id_map[x])
        #get edge index
        self.edge_index = self.edges[['source','target']].values.T
        self.feats.set_index(['id'],inplace=True)
        self.labels.set_index(['id'],inplace=True)
        numerical_feats = ['followers_count','following_count','active_days','name_len','uname_len','tweet_count','listed_count']
        category_feats = ['protected','verified','picture']
        num_property_embedding = torch.FloatTensor(self.feats[numerical_feats].values)
        cat_property_embedding = torch.FloatTensor(self.feats[category_feats].values)
        x = torch.cat([num_property_embedding, cat_property_embedding], dim=1)
        y = torch.LongTensor(self.labels['label'].tolist())
        edge_index = torch.LongTensor(self.edge_index)
        graphlet_feat = torch.FloatTensor(np.load(self.root+'/'+self.raw_file_names[3]))
        data_list = []
        graph = Data(edge_index=edge_index,
                    y=y,
                    x=x,
                    num_nodes=num_property_embedding.shape[0],
                    x_graphlet = graphlet_feat)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def MultiSrcAndTarLoader(tgt_idx:int, src_idx:list,batch_size, device):
    """

    :param tgt_idx:
    :param src_idx:
    :param batch_size:
    :param device:
    :return:
    multi-src and target idx loader & full graphObj
    """
    tgt_ds = Twibot22DomainData(graph_idx=tgt_idx)[0]
    src_ds_bank = [Twibot22DomainData(graph_idx=x)[0] for x in src_idx]
    # batch nodes with node id
    tgt_dl = DataLoader(TensorDataset(torch.arange(0,tgt_ds.x.size(0))), batch_size=batch_size, shuffle=True, drop_last=True)
    src_dl_bank = [DataLoader(TensorDataset(torch.arange(0,ds.x.size(0))), batch_size=batch_size, shuffle=True, drop_last=True) for ds in src_ds_bank]
    src_dl_bank.append(tgt_dl)
    n_nodes = [ds.x.size(0) for ds in src_ds_bank]+[tgt_ds.x.size(0)]
    max_node_idx = n_nodes.index(max(n_nodes))
    src_dl_bank = [cycle(dl) if i!=max_node_idx else dl for i,dl in enumerate(src_dl_bank) ]
    full_idx_loader = src_dl_bank
    # full_loader = enumerate(zip(*src_dl_bank))
    full_data = [data.to(device) for data in src_ds_bank]+[tgt_ds.to(device)]
    return full_idx_loader, full_data

def MixMultiSrcAndTarLoader(tgt_idx:int, src_idx:list,src_batch_size, tgt_batch_size, device):
    """
    :param tgt_idx:
    :param src_idx:
    :param batch_size:
    :param device:
    :return:
    multi-src and target idx loader & full graphObj
    """
    tgt_ds = Twibot22DomainData(graph_idx=tgt_idx)[0]
    src_ds_bank = [Twibot22DomainData(graph_idx=x)[0] for x in src_idx]
    # batch nodes with node id
    tgt_dl = DataLoader(TensorDataset(torch.arange(0,tgt_ds.x.size(0))), batch_size=tgt_batch_size, shuffle=True, drop_last=True)
    src_dl_bank = [DataLoader(TensorDataset(torch.arange(0,ds.x.size(0))), batch_size=src_batch_size, shuffle=True, drop_last=True) for ds in src_ds_bank]
    src_dl_bank.append(tgt_dl)
    n_nodes = [ds.x.size(0) for ds in src_ds_bank]
    max_node_idx = n_nodes.index(max(n_nodes))
    src_dl_bank = [cycle(dl) if i!=max_node_idx else dl for i,dl in enumerate(src_dl_bank) ]
    full_idx_loader = src_dl_bank
    # full_loader = enumerate(zip(*src_dl_bank))
    full_data = [data.to(device) for data in src_ds_bank]+[tgt_ds.to(device)]
    return full_idx_loader, full_data





