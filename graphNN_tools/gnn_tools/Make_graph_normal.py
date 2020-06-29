import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset  
import pickle
import os
import os.path as osp

class Make_graph_normal(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Make_graph_normal, self).__init__(root, transform, pre_transform)
        df_reduced = pd.read_pickle("../data/df_reduced2.pic")
        self.dataLength = len(df_reduced)
        
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']

    def download(self):
        pass
    
    def process(self):
        df_reduced = pd.read_pickle("../data/df_reduced2.pic")
        df_mbtr = pd.read_pickle("../data/mbtr1.pic")
        MBTR_batch_size = len(df_mbtr)
        counter = 0
        for MBTR_batch in range(0,len(df_reduced), MBTR_batch_size):
            counter += 1
            print("Batch =", counter)
            if counter > 1:
                df_mbtr = pd.read_pickle("../data/mbtr%s.pic"%counter)
            df_mbtr = df_mbtr[['mbtr']]               
            for i, row in df_mbtr.iterrows():
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
                
                # Get MBTR features
                mbtr = row["mbtr"]
                mbtr = torch.tensor(mbtr, dtype=torch.float)

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

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u, mbtr=mbtr)
                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def __len__(self):
        return self.dataLength

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
        
