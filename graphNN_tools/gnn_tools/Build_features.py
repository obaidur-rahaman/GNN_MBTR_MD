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
from rdkit.Chem.rdchem import HybridizationType
import os
from rdkit.Chem import ChemicalFeatures, rdMolChemicalFeatures, Descriptors, rdMolDescriptors as rdm
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory
from rdkit.Chem.AllChem import EmbedMolecule
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit import RDConfig
from sklearn import preprocessing
from dscribe.descriptors import MBTR
import collections
from PyAstronomy import pyasl
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase import Atoms
import re
import os.path as osp

class Build_features():
    def __init__(self, df, addH1, XYZ1, nbr_Gaussian1, NB_cutoff1):
        global addH, XYZ, nbr_Gaussian, NB_cutoff
        self.df = df
        addH = addH1
        atomicNbrs, elements, values = self.getAtomicDist(self.df)
        self.atomicNbrs = atomicNbrs
        self.elements = elements 
        self.x_mean = []
        self.edge_attr_mean = []     
        XYZ = XYZ1
        nbr_Gaussian = nbr_Gaussian1
        NB_cutoff = NB_cutoff1

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
            if (0 == XYZ):
                mol = Chem.AddHs(mol) 
                EmbedMolecule(mol)
            if (0 == addH):
                mol = Chem.RemoveHs(mol)
            else:
                mol = Chem.AddHs(mol)
            print("Getting features for mol =", i, "refcode =", df.at[i,"refcode_csd"])
            df.at[i,"x"], x_sum, x_nbrAtoms = self.get_node_features(mol)
            if (1 == XYZ):
                df.at[i,"edge_index"],  df.at[i,"edge_attr"], edge_attr_sum, edge_attr_nbrBonds = self.get_edge_features(mol, i)
            else:
                df.at[i,"edge_index"],  df.at[i,"edge_attr"], df.at[i,"xyz"], edge_attr_sum, edge_attr_nbrBonds = self.get_edge_features(mol, i)
            df.at[i,"u"] = self.get_global_features(mol)
            x_sum_all = x_sum
            x_nbrAtoms_all = x_nbrAtoms
            edge_attr_sum_all = edge_attr_sum
            edge_attr_nbrBonds_all = edge_attr_nbrBonds
            if 0 != len(df.at[i,"edge_attr"]) :
                if (0 == i):
                    x_sum_all = x_sum
                    x_nbrAtoms_all = x_nbrAtoms
                    edge_attr_sum_all = edge_attr_sum
                    edge_attr_nbrBonds_all = edge_attr_nbrBonds
                else:
                    x_sum_all = [x_sum_all[i]+x_sum[i] for i in range(len(x_sum))]
                    x_nbrAtoms_all = x_nbrAtoms_all + x_nbrAtoms
                    edge_attr_sum_all = [edge_attr_sum_all[i]+edge_attr_sum[i] for i in range(len(edge_attr_sum))]
                    edge_attr_nbrBonds_all = edge_attr_nbrBonds_all + edge_attr_nbrBonds
        self.x_mean = [x_sum_all[i]/x_nbrAtoms_all for i in range(len(x_sum_all))] 
        self.edge_attr_mean = [edge_attr_sum_all[i]/edge_attr_nbrBonds_all for i in range(len(edge_attr_sum_all))]
        #print("Mean values of atomic features", self.x_mean)
        #print("Mean values of bond features", self.edge_attr_mean)
        # Now drop the rows that are marked with 'drop'
        old_len = len(df)
        df = df[df.edge_index != 'drop']
        df = df.reset_index(drop=True)
        new_len = len(df)
        if (old_len != new_len):
            dropped = old_len - new_len
            print(dropped, "rows were deleted because the coordinates in mol and XYZ did not have the same order")
        self.df = df
        return(df)
    
    def get_node_features(self,mol):  
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)  
        x = []
        sp = []
        sp2 = []
        sp3 = []
        donor = []
        acceptor = []
        type_idx = []
        for i in range(4):
            x.append([])
        for atom in mol.GetAtoms(): 
            donor.append(0)
            acceptor.append(0)
            #print(atom.GetHybridization())     
            x[0].append(atom.GetAtomicNum())
            x[1].append(atom.GetTotalValence())
            x[2].append(atom.GetIsAromatic())
            x[3].append(atom.GetTotalNumHs(includeNeighbors=True))
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            type_idx.append(atom.GetAtomicNum())
        # Now calculate donors and acceptors
        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1
        # Now get the elements and make one hot encoding
        x_element = []
        for i in range(len(self.atomicNbrs)):
            x_element.append([])
        for i in range(len(type_idx)):
            for j in range(len(self.atomicNbrs)):
                if type_idx[i] == self.atomicNbrs[j]:
                    x_element[j].append(1)
                else:
                    x_element[j].append(0)
        # Now add them all 
        x = x + [donor] + [acceptor] + [sp] + [sp2] + [sp3] + x_element 
        x = np.array(x).T
        #x = torch.tensor(x, dtype=torch.float)
        return(x, x.sum(axis=0), len(x))
    '''   
    mol = Chem.MolFromSmiles('Clc1ccccc1Cl')
    mol = Chem.MolFromSmiles('C#C')
    '''
    def get_edge_features(self,mol,molNbr):
       
        if (1 == XYZ):
            # First get the spatial info from xyz
            xyz = self.df.loc[molNbr:molNbr,'xyz'].values[0]

            totNbrAtoms = re.findall("^\d+\n", xyz)[0][:-1]
            #print("totNbrAtoms =", totNbrAtoms)
            atomType = []
            coordinates = []
            for match in re.finditer('[A-Z][a-z]?\s+\-?\d+\.(\d+)?\s+\-?\d+\.(\d+)?\s+\-?\d+\.(\d+)?', xyz):
                s = match.start()
                e = match.end()
                data = xyz[s:e]
                atomType.append(data.split()[0])
                coordinates.append((data.split()[1],data.split()[2], data.split()[3] ))
            #print(coordinates)

            coordinates = np.array(coordinates)
            coordinates = coordinates.astype(np.float)

            # Do some sanity checks
            # Check if all atoms are read from XYZ
            if (len(coordinates) != int(totNbrAtoms)):
                print("Not all atoms could be extracted from XYZ. Dropping from Dataset")
                return('drop', [], [], [])
            # Check if the atomic sequence is the same in mol and XYZ
            for i in range(len(atomType)):
                if atomType[i] != mol.GetAtoms()[i].GetSymbol():
                    print("Atomic order in mol and XYZ are not matching. Dropping from Dataset")
                    return('drop', [], [], [])
        else:
            molMatrix = GetMoleculeBoundsMatrix(mol)

        ind1 = []
        ind2 = []
        edge_attr = []

        # BONDED INTERACTIONS
        type_idx = []
        
        for i in range(nbr_Gaussian + 1):
            edge_attr.append([])       
        for bondNbr in range(mol.GetNumBonds()):
            #print("bond number =",bond)
            atom1 = mol.GetBondWithIdx(bondNbr).GetBeginAtomIdx()
            atom2 = mol.GetBondWithIdx(bondNbr).GetEndAtomIdx()
            #print(atom1, atom2)
            #print(atom1, atom2, mol.GetAtomWithIdx(atom1).GetSymbol(), mol.GetAtomWithIdx(atom2).GetSymbol())
            ind1.append(atom1)
            ind1.append(atom2)
            ind2.append(atom2)
            ind2.append(atom1)
            bond = mol.GetBonds()[bondNbr]

            type_idx.append(bond.GetBondTypeAsDouble())
            type_idx.append(bond.GetBondTypeAsDouble())

            if (0 == XYZ):
                avg_bond_length = (molMatrix[atom1][atom2] + molMatrix[atom2][atom1])/2
            else:
                avg_bond_length = np.sqrt( (coordinates[atom1][0] - coordinates[atom2][0])**2 +  (coordinates[atom1][1] - coordinates[atom2][1])**2 + (coordinates[atom1][2] - coordinates[atom2][2])**2)
            coulomb = (mol.GetAtoms()[atom1].GetAtomicNum() * mol.GetAtoms()[atom2].GetAtomicNum()) / avg_bond_length
            avg_bond_length = self.Gaussian_distance(avg_bond_length, nbr_Gaussian)
            for attr_nbr in range(nbr_Gaussian):
                edge_attr[attr_nbr].append(avg_bond_length[attr_nbr])
                edge_attr[attr_nbr].append(avg_bond_length[attr_nbr])

            edge_attr[nbr_Gaussian].append(coulomb)
            edge_attr[nbr_Gaussian].append(coulomb)

        ind1_bonded = ind1
        ind2_bonded = ind2
        # NON BONDED INTERACTIONS
        for atom1 in range(len(GetMoleculeBoundsMatrix(mol))):
            for atom2 in range(len(GetMoleculeBoundsMatrix(mol))):
                # First check if this pair is ABSENT in the bonded pair list 
                flag = 0               
                for i in range(len(ind1_bonded)):
                    if (atom1 == ind1_bonded[i]) and (atom2 == ind2_bonded[i]):
                       flag = 1
                if (0 == flag): 
                    #print(atom1,atom2)
                    if (0 == XYZ):
                        avg_dist = (molMatrix[atom1][atom2] + molMatrix[atom2][atom1])/2
                    else:
                        avg_dist = np.sqrt( (coordinates[atom1][0] - coordinates[atom2][0])**2 +  (coordinates[atom1][1] - coordinates[atom2][1])**2 + (coordinates[atom1][2] - coordinates[atom2][2])**2)
                    
                    if (0.0 < avg_dist) and (avg_dist < NB_cutoff):
                        #print(avg_dist)
                        ind1.append(atom1)
                        ind1.append(atom2)
                        ind2.append(atom2)
                        ind2.append(atom1)

                        type_idx.append(0.0)
                        type_idx.append(0.0)

                        coulomb = (mol.GetAtoms()[atom1].GetAtomicNum() * mol.GetAtoms()[atom2].GetAtomicNum()) / avg_dist
                        avg_dist = self.Gaussian_distance(avg_dist, nbr_Gaussian)
                        for attr_nbr in range(nbr_Gaussian):
                            edge_attr[attr_nbr].append(avg_dist[attr_nbr])
                            edge_attr[attr_nbr].append(avg_dist[attr_nbr])

                        edge_attr[nbr_Gaussian].append(coulomb)
                        edge_attr[nbr_Gaussian].append(coulomb)
        '''
        for atom in mol.GetAtoms(): 
            #print(atom.GetHybridization())     
            print(atom.GetAtomicNum())    
            GetMoleculeBoundsMatrix(mol)[6][7]
        '''

        # Now get the bond_type and make one hot encoding
        bond_type = []
        for i in range(5):
            bond_type.append([])
        for i in range(len(type_idx)):
            for j,btype in enumerate([0.0, 1.0, 1.5, 2.0, 3.0]):
                #print(j, btype)
                if type_idx[i] == btype:
                    bond_type[j].append(1)
                else:
                    bond_type[j].append(0)

        ind1 = np.array(ind1)  
        ind2 = np.array(ind2)
        edge_index = torch.tensor([ind1,ind2], dtype=torch.long)
        edge_attr =  bond_type + edge_attr
        edge_attr = np.array(edge_attr).T
        if (1 == XYZ):
            return(edge_index, edge_attr, edge_attr.sum(axis=0), len(edge_attr))
        else:
            xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
            #print(xyz)
            return(edge_index, edge_attr, xyz, edge_attr.sum(axis=0), len(edge_attr))

    def Gaussian_distance(self, dist, nbr_Gaussian):
        mu = np.linspace(0, 5, nbr_Gaussian)
        dist = np.exp(-(dist - mu) ** 2 / (0.5 ** 2))
        return(dist)


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
#        FpDensityMorgan1 = Descriptors.FpDensityMorgan1(mol)
#        FpDensityMorgan2 = Descriptors.FpDensityMorgan2(mol)
#        FpDensityMorgan3 = Descriptors.FpDensityMorgan3(mol)
        
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
        
        
        u = [natoms, nbonds, mw, HeavyAtomMolWt, NumValenceElectrons, \
            nbrAcceptor, nbrDonor, nbrHydrophobe, nbrLumpedHydrophobe, \
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
    def normalize_features(self, feature):
        # node features and edge attributes
        #for feature in ['x','edge_attr']:
        if ('x' == feature) or ('edge_attr' == feature):
            # We already determined the mean values while calculating the features
            # Let's calculate the standard deviations
            print("Normalizing", feature,": step 1 = get std")
            if ('x' == feature):
                mean_values = self.x_mean
            else:
                mean_values = self.edge_attr_mean
            nbr_features = len(self.df.loc[0:0,feature][0][0])
            sum_squares = []
            for i in range(nbr_features):
                sum_squares.append(0)
            nbr = 0
            for i, row in self.df.iterrows():
                for k in range(len(row[feature])):
                    nbr = nbr + 1
                    values_list = row[feature][k]
                    sum_squares = [sum_squares[i] + (values_list[i] - mean_values[i])**2 for i in range(len(values_list))]
            std_values = [np.sqrt(sum_squares[i] / nbr) for i in range(len(sum_squares))]
            #print("std values for", feature, std_values)
            
            # Now normalize
            print("Normalizing", feature,": step 2 = scale")
            for i, row in self.df.iterrows():
                scaled_values_all = []
                # Do this for all the atoms in the molecule
                for k in range(len(row[feature])):
                    values_list = row[feature][k]                   
                    scaled_values = [(values_list[i] - mean_values[i])/std_values[i] for i in range(len(values_list))]                   
                    scaled_values_all.append(scaled_values)
                #print("-----------------------")
                #print(scaled_values_all)
                scaled_values_all = np.array(scaled_values_all)
                #scaled_values_all = sparse.COO(scaled_values_all) # Makes it very slow for the next step
                self.df.at[i,feature] = scaled_values_all

        # Now normalize the global features and MBTR features
        #for descriptor in ["u"]: #,"mbtr"]:
        if ('u' == feature) or ('mbtr' == feature):
            print("Normalizing", feature)
            df1 = pd.DataFrame(self.df[feature].values.tolist()) # Create a new df with the elements of the list as columns
            x = df1.values
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(x)
            x_scaled = min_max_scaler.transform(x)
            for i in range(len(x_scaled)):
                self.df.at[i,feature] = x_scaled[i]  
        return(self.df)

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
        for match in re.finditer('[A-Z][a-z]?\s+\-?\d+\.(\d+)?\s+\-?\d+\.(\d+)?\s+\-?\d+\.(\d+)?', xyz):
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
        #mbtr_output[mbtr_output == 0] = ['nan']
        for i in range(mbtr_output.shape[0]):
            mbtr_output[i] = round(float(mbtr_output[i]),8)
        return(mbtr_output)


    def getAtomicDist(self,df):
        all_atoms = []
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            mol = Chem.MolFromSmiles(df.loc[i,"smiles"])
            #print("i =", i, "smile =", df.loc[i,"smiles"], "mol =", mol)
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

    def get_atomic_distribution(self, mol):
        if (1 == addH):
            mol = Chem.AddHs(mol)    
        x = []
        for atom in mol.GetAtoms():      
            x.append(atom.GetAtomicNum())    
        return(x)
