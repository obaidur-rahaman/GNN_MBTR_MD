import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)
import pickle
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import os
from rdkit.Chem import ChemicalFeatures, rdMolChemicalFeatures, Descriptors, rdMolDescriptors as rdm
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory
from rdkit import RDConfig
import os.path as osp

class Make_graph_inMemory(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Make_graph_inMemory, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        df_reduced = pd.read_pickle("../data/df_reduced2.pic")              
        for i in range(len(df_reduced)):
            name = df_reduced.at[i,"refcode_csd"]       
            print(i, name)
            # Now extract the details to construct the graph               
            # Get node features               
            x = df_reduced.at[i,"x"]
            x = torch.tensor(x, dtype=torch.float)
            # Get edge indices and features/attributes              
            edge_index = df_reduced.at[i,"edge_index"]                
            edge_attr = df_reduced.at[i,"edge_attr"]
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Get global features               
            u = df_reduced.at[i,"u"]
            u = torch.tensor(u, dtype=torch.float)

            # Get all the targets              
            y = []
                
            file1 = open('../data/target_terms_all.txt', 'r') 
            Lines = file1.readlines() 
            for target_term in Lines: 
                target_term = target_term.strip()
                #print(target_term)
                y.append(df_reduced.loc[i,target_term])
            
            y = np.array(y).T
            y = torch.tensor(y, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
            data_list.append(data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
