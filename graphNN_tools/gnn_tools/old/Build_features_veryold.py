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
from sklearn import preprocessing
from dscribe.descriptors import MBTR
import collections
from PyAstronomy import pyasl
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase import Atoms
import re
import tracemalloc
import gc

class Build_features:
    def __init__(self):
        pass

    def get_all_features(self, df, mol_list):

        # NOW GET ALL THE FEATURES
        df = df.astype('object')
        df['x'] = ''
        df['edge_index'] = ''
        df['edge_attr'] = ''
        df['u'] = ''

        print("Length of df =", len(df), "Length of mol_list =", len(mol_list))
        for i in range(len(mol_list)):
            mol = mol_list[i]
            print("Getting features for mol =", i, "refcode =", df.at[i,"refcode_csd"])
            df.at[i,"x"] = self.get_node_features(mol)
            df.at[i,"edge_index"],  df.at[i,"edge_attr"] = self.get_edge_features(mol)
            df.at[i,"u"] = self.get_global_features(mol)  
        return(df)

    def get_MBTR_feature(self, df, mbtr_constructor):

        # NOW GET MBTR FEATURE
        df = df.astype('object')
        df['mbtr'] = ''      
        print("Length of df =", len(df))
        for i, row in df.iterrows():
            print("Getting MBTR feature for mol =", i)
            df.at[i,"mbtr"] = self.get_MBTR(mbtr_constructor, row["xyz"])  
        df = df.drop(['xyz'], axis=1)
        return(df)

    def normalize_features(self,df, feature):
        # node features and edge attributes
        #for feature in ['x','edge_attr']:
        if ('x' == feature) or ('edge_attr' == feature):
        # First determine the mini and maxi for each feature
            df1 = pd.DataFrame(df[feature].values.tolist())
            df1 = df1.astype('object')
            df1['new'] = ''
            mini = []
            maxi = []
            print("Normalizing", feature,": step 1 = find mini and maxi")
            for i in range(len(df1)):
                node_features = df1.loc[i,0]
                if (0 == i):
                    for nbr in range(len(node_features[0])):
                        #print(nbr)
                        mini.append(node_features[0][nbr])
                        maxi.append(node_features[0][nbr])
                
                for j in range(len(node_features)):
                    #print("node_feature =",node_features[j])
                    for k in range(len(node_features[j])):
                        feature_value = node_features[j][k]
                        if (feature_value < mini[k]):
                            mini[k] = feature_value
                        if (feature_value > maxi[k]):
                            maxi[k] = feature_value
            # then normalize
            print("Normalizing", feature,": step 2 = scale")
            for i in range(len(df1)):
                #print(i)
                node_features = df1.loc[i,0] 
                all_list = []
                for j in range(len(node_features)):
                    temp_list = []
                    for k in range(len(node_features[j])):
                        feature_value = node_features[j][k]
                        if (mini[k] != maxi[k]):
                            new_value = (feature_value - mini[k]) / (maxi[k] - mini[k])
                            temp_list.append(new_value)
                        else:
                            temp_list.append(feature_value)
                    all_list.append(temp_list) 
                df1.at[i,"new"] = all_list              
            df[feature] = df1['new']
        # Now normalize the global features and MBTI features
        #for descriptor in ["u"]: #,"mbtr"]:
        if ('u' == feature) or ('mbtr' == feature):
            print("Normalizing", feature)
            df1 = pd.DataFrame(df[feature].values.tolist()) # Create a new df with the elements of the list as columns
            x = df1.values
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(x)
            x_scaled = min_max_scaler.transform(x)
            for i in range(len(x_scaled)):
                df.at[i,feature] = x_scaled[i]  
        return(df)
    
    def get_node_features(self,mol):    
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
        #x = torch.tensor(x, dtype=torch.float)
        return(x)
    '''   
    mol = Chem.MolFromSmiles('Clc1ccccc1Cl')
    '''
    def get_edge_features(self,mol):
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
        #edge_attr = torch.FloatTensor(edge_attr)
        #edge_attr = edge_attr.to(torch.float32)
        return(edge_index, edge_attr)
        
    def get_global_features(self,mol):
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

        #u = torch.tensor(u, dtype=torch.float)
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
    def get_MBTR(self,mbtr, xyz):
        # Pubchem does not like \t\n etc characters at the end of smile
        # Lets remove them
        '''
        print("smile initial =", smile)
        if ("\n" == smile[-1]):
            smile = smile[:-1]
        if ("\t" == smile[-1]):
            smile = smile[:-1]
        print("smile final =", smile)

        #mol_MBTR = pubchem_atoms_search(smiles=smile)
        '''

        # Now read in the atomic positions 

        totNbrAtoms = re.findall("^\d+\n", xyz)[0][:-1]
        #print("totNbrAtoms =", totNbrAtoms)
        atomType = []
        coordinates = []
        for match in re.finditer('\w+\s+\-?\d+\.\d+\s+\-?\d+\.\d+\s+\-?\d+\.\d+', xyz):
            s = match.start()
            e = match.end()
            data = xyz[s:e]
            atomType.append(data.split()[0])
            coordinates.append((data.split()[1],data.split()[2], data.split()[3] ))
            #print(xyz[s:e])

        my_atoms = Atoms('N' + totNbrAtoms, coordinates)
        my_atoms.set_chemical_symbols(atomType)

        # Create MBTR output for the system
        mbtr_output = mbtr.create(my_atoms)
        #print("mbtr_output =",mbtr_output)
        mbtr_output = mbtr_output[0]
        # Convert all zeros to NAN
        mbtr_output[mbtr_output == 0] = ['nan']
        for i in range(mbtr_output.shape[0]):
            mbtr_output[i] = round(float(mbtr_output[i]),8)
        return(mbtr_output)


    def getAtomicDist(self,df):
        all_atoms = []
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            #print("i =", i)
            mol = Chem.MolFromSmiles(df.loc[i,"smiles"])
            value = self.get_atomic_distribution(mol)
            for j in range(len(value)):
                all_atoms.append(value[j])
        
        all_atoms.sort()
        
        counter=collections.Counter(all_atoms)
        #print(counter)
        #print(counter.values())
        #print(counter.keys())
        
        an = pyasl.AtomicNo()
        #an.showAll()
        elements = []
        atomicNbrs = []
        for i in range(len(counter.keys())):
            elements.append(an.getElSymbol(list(counter.keys())[i]))
            atomicNbrs.append(list(counter.keys())[i])
        return(atomicNbrs, elements, counter.values())

    def get_atomic_distribution(self,mol):
        mol = Chem.AddHs(mol)    
        x = []
        for atom in mol.GetAtoms():      
            x.append(atom.GetAtomicNum())    
        return(x)

