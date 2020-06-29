import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)
import pickle
from sklearn.preprocessing import StandardScaler

class Make_graph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Make_graph, self).__init__(root, transform, pre_transform)
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
        df_reduced = pd.read_pickle("../data/dataframe_reduced.pic")
        
        with open("../data/mol_list", "rb") as fp:   # Unpickling
            mol_list= pickle.load(fp)
            
        for i in range(len(mol_list)):
            mol = mol_list[i]
            name = df_reduced.at[i,"refcode_csd"]       
            print(i, name)
            # Now extract the details to construct the graph
            
            # Get node features
            
            x = []
            for atom in mol.GetAtoms():      
               x.append(atom.GetAtomicNum())
            x = list(map(int, x))
            x = torch.tensor(x, dtype=torch.float)
           
            # Get edge indices and features/attributes
            ind1 = []
            ind2 = []
            bondType = []        
            for bond in range(mol.GetNumBonds()):
                #print("bond number =",bond)
                atom1 = mol.GetBondWithIdx(bond).GetBeginAtomIdx()
                atom2 = mol.GetBondWithIdx(bond).GetEndAtomIdx()
                type = mol.GetBonds()[bond].GetBondTypeAsDouble()
                #print(atom1, atom2, mol.GetAtomWithIdx(atom1).GetSymbol(), mol.GetAtomWithIdx(atom2).GetSymbol())
                ind1.append(atom1)
                ind2.append(atom2)
                bondType.append(type)
            
            ind1 = np.array(ind1)  
            ind2 = np.array(ind2)
            edge_index = torch.tensor([ind1,ind2], dtype=torch.long)
            bondType = torch.FloatTensor(bondType)
            edge_attr = bondType.to(torch.float32)

            with open("../data/target_term.txt","r") as file:
                target_term = file.read() 

            y = df_reduced.at[i,target_term]
            #y = y /1000
            y =  np.log10(y)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        
        #print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


        
