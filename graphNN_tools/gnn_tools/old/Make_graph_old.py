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
            
            # Get global features
            
            u = get_global_features(mol) 
            
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
    
def get_global_features(mol):
    u = []       
    # Now get some specific features
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    # First get some basic features
    natoms = mol.GetNumAtoms()
    nbonds = mol.GetNumBonds()
    mw = Descriptors.ExactMolWt(mol)
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    ''' # These four descriptors are producing the value of infinity for refcode_csd = YOLJUF (CCOP(=O)(Cc1ccc(cc1)NC(=S)NP(OC(C)C)(OC(C)C)[S])OCC\t\n)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MaxPartialCharge = Descriptors.MaxPartialCharge(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    MinPartialCharge = Descriptors.MinPartialCharge(mol)
    '''
    FpDensityMorgan1 = Descriptors.FpDensityMorgan1(mol)
    FpDensityMorgan2 = Descriptors.FpDensityMorgan2(mol)
    FpDensityMorgan3 = Descriptors.FpDensityMorgan3(mol)
     
    # Get some features using chemical feature factory
    
    nbrAcceptor = 0
    nbrDonor = 0
    nbrHydrophobe = 0
    nbrLumpedHydrophobe = 0
    nbrPosIonizable = 0
    nbrNegIonizable = 0
 
    for j in range(len(feats)):
        #print(feats[j].GetFamily(), feats[j].GetType())
        if ('Acceptor' == (feats[j].GetFamily())):
            nbrAcceptor = nbrAcceptor + 1
        elif ('Donor' == (feats[j].GetFamily())):
            nbrDonor = nbrDonor + 1
        elif ('Hydrophobe' == (feats[j].GetFamily())):
            nbrHydrophobe = nbrHydrophobe + 1
        elif ('LumpedHydrophobe' == (feats[j].GetFamily())):
            nbrLumpedHydrophobe = nbrLumpedHydrophobe + 1
        elif ('PosIonizable' == (feats[j].GetFamily())):
            nbrPosIonizable = nbrPosIonizable + 1
        elif ('NegIonizable' == (feats[j].GetFamily())):
            nbrNegIonizable = nbrNegIonizable + 1                 
        else:
            pass
            #print(feats[j].GetFamily())
    
    # Now get some features using rdMolDescriptors
    
    moreGlobalFeatures = [rdm.CalcNumRotatableBonds(mol), rdm.CalcChi0n(mol), rdm.CalcChi0v(mol), \
                          rdm.CalcChi1n(mol), rdm.CalcChi1v(mol), rdm.CalcChi2n(mol), rdm.CalcChi2v(mol), \
                          rdm.CalcChi3n(mol), rdm.CalcChi4n(mol), rdm.CalcChi4v(mol), \
                          rdm.CalcFractionCSP3(mol), rdm.CalcHallKierAlpha(mol), rdm.CalcKappa1(mol), \
                          rdm.CalcKappa2(mol), rdm.CalcLabuteASA(mol), \
                          rdm.CalcNumAliphaticCarbocycles(mol), rdm.CalcNumAliphaticHeterocycles(mol), \
                          rdm.CalcNumAliphaticRings(mol), rdm.CalcNumAmideBonds(mol), \
                          rdm.CalcNumAromaticCarbocycles(mol), rdm.CalcNumAromaticHeterocycles(mol), \
                          rdm.CalcNumAromaticRings(mol), rdm.CalcNumBridgeheadAtoms(mol), rdm.CalcNumHBA(mol), \
                          rdm.CalcNumHBD(mol), rdm.CalcNumHeteroatoms(mol), rdm.CalcNumHeterocycles(mol), \
                          rdm.CalcNumLipinskiHBA(mol), rdm.CalcNumLipinskiHBD(mol), rdm.CalcNumRings(mol), \
                          rdm.CalcNumSaturatedCarbocycles(mol), rdm.CalcNumSaturatedHeterocycles(mol), \
                          rdm.CalcNumSaturatedRings(mol), rdm.CalcNumSpiroAtoms(mol), rdm.CalcTPSA(mol)]
    
    
    u = [natoms, nbonds, mw, HeavyAtomMolWt, NumValenceElectrons, FpDensityMorgan1, FpDensityMorgan2, \
         FpDensityMorgan3, nbrAcceptor, nbrDonor, nbrHydrophobe, nbrLumpedHydrophobe, \
         nbrPosIonizable, nbrNegIonizable]

    u = u + moreGlobalFeatures    
    u = np.array(u).T
    # Some of the descriptors produice NAN. We can convert them to 0
    # If you are getting outliers in the training or validation set this could be 
    # Because some important features were set to zero here because it produced NAN
    # Removing those features from the feature set might remove the outliers
    
    #u[np.isnan(u)] = 0

    u = torch.tensor(u, dtype=torch.float)
    return(u)
'''    
# Determine which side chains the molecule has
import rdkit    
mol = Chem.MolFromSmiles('CCOC(=O)C1(C#N)C(c2ccccc2)C1(C#N)c1ccc(Cl)cc1')
mol2 = Chem.MolFromSmiles('Clc1ccccc1Cl')
from rdkit.Chem.rdmolops import ReplaceCore
from rdkit.Chem.Scaffolds import MurckoScaffold
core = core = MurckoScaffold.GetScaffoldForMol(mol)
list_side = rdkit.Chem.rdmolops.ReplaceCore(mol, core)


# Determine other features
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
mol = Chem.MolFromSmiles('CCOP(=O)(Cc1ccc(cc1)NC(=S)NP(OC(C)C)(OC(C)C)[S])OCC\t\n')
Descriptors.MaxAbsPartialCharge(mol, force=False)
