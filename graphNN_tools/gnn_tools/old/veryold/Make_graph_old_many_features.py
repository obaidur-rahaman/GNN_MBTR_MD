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
        df_reduced = pd.read_pickle("../data/df_reduced.pic")
        
        with open("../data/mol_list", "rb") as fp:   # Unpickling
            mol_list= pickle.load(fp)
            
        for i in range(len(mol_list)):
            mol = mol_list[i]
            name = df_reduced.at[i,"refcode_csd"]       
            print(i, name)
            # Now extract the details to construct the graph
            
            # Get node features
            
            x = get_node_features(mol)
           
            # Get edge indices and features/attributes
            
            edge_index, edge_attr = get_edge_features(mol)

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

def get_node_features(mol):    
    x = []
    for i in range(12):
        x.append([])
    for atom in mol.GetAtoms():      
        x[0].append(atom.GetAtomicNum())
        x[1].append(atom.GetMass())
        x[2].append(atom.GetTotalValence())
        x[3].append(atom.GetNumImplicitHs())
        x[4].append(atom.GetFormalCharge())
        x[5].append(atom.GetNumRadicalElectrons())
        x[6].append(atom.GetImplicitValence())
        x[7].append(atom.GetNumExplicitHs())
        x[8].append(atom.GetIsAromatic())
        x[9].append(atom.GetIsotope())
        x[10].append(atom.GetChiralTag())
        x[11].append(atom.GetHybridization())
       
    x = np.array(x).T
    x = torch.tensor(x, dtype=torch.float)
    return(x)
'''   
mol = Chem.MolFromSmiles('Clc1ccccc1Cl')
'''
def get_edge_features(mol):
    ind1 = []
    ind2 = []
    edge_attr = [] 
    for i in range(7):
        edge_attr.append([])       
    for bondNbr in range(mol.GetNumBonds()):
        #print("bond number =",bond)
        atom1 = mol.GetBondWithIdx(bondNbr).GetBeginAtomIdx()
        atom2 = mol.GetBondWithIdx(bondNbr).GetEndAtomIdx()
        #print(atom1, atom2, mol.GetAtomWithIdx(atom1).GetSymbol(), mol.GetAtomWithIdx(atom2).GetSymbol())
        ind1.append(atom1)
        ind2.append(atom2)
        bond = mol.GetBonds()[bondNbr]
        edge_attr[0].append(bond.GetBondTypeAsDouble())
        edge_attr[1].append(bond.GetIsAromatic())
        edge_attr[2].append(bond.GetIsConjugated())
        bond_type = bond.GetBondTypeAsDouble()
        if (bond_type == 1):
            edge_attr[3].append(1)
        else:
            edge_attr[3].append(0)
        if (bond_type == 1.5):
            edge_attr[4].append(1)
        else:
            edge_attr[4].append(0)
        if (bond_type == 2):
            edge_attr[5].append(1)
        else:
            edge_attr[5].append(0)
        if (bond_type == 3):
            edge_attr[6].append(1)
        else:
            edge_attr[6].append(0)
            
    ind1 = np.array(ind1)  
    ind2 = np.array(ind2)
    edge_index = torch.tensor([ind1,ind2], dtype=torch.long)
    edge_attr = np.array(edge_attr).T
    edge_attr = torch.FloatTensor(edge_attr)
    edge_attr = edge_attr.to(torch.float32)
    return(edge_index, edge_attr)
'''    
import rdkit    
mol = Chem.MolFromSmiles('CCOC(=O)C1(C#N)C(c2ccccc2)C1(C#N)c1ccc(Cl)cc1')
mol2 = Chem.MolFromSmiles('Clc1ccccc1Cl')
from rdkit.Chem.rdmolops import ReplaceCore
from rdkit.Chem.Scaffolds import MurckoScaffold
core = core = MurckoScaffold.GetScaffoldForMol(mol)
list_side = rdkit.Chem.rdmolops.ReplaceCore(mol, core)


from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, rdMolChemicalFeatures
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory
from rdkit import RDConfig
import os
fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
feats = factory.GetFeaturesForMol(mol)
feats2 = factory.GetFeaturesForMol(mol2)

for i in range(len(feats)):
    print(feats[i].GetFamily(), feats[i].GetType())
    if ('Acceptor' == feats[i].GetFamily()):
        print("match")
'''