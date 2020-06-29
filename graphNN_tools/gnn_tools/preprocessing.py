import pandas as pd
import numpy as np
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import os
from rdkit.Chem import ChemicalFeatures, rdMolChemicalFeatures
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory
from rdkit import RDConfig
from rdkit.Chem.AllChem import EmbedMolecule
from sklearn.utils import shuffle
import pickle 
import subprocess
import gnn_tools as gnn
import time
import re

def preprocessData(df, end): 
         
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df[:end]    
    df_reduced, mol_list, legends = get_molecules(df)
    print("Size of original dataframe =",len(df))
    print("Size of reduced dataframe =", len(df_reduced), "(excluding the cases that could not be processed)")   
    with open("../data/mol_list", "wb") as fp:
        pickle.dump(mol_list, fp)
    return(df_reduced, mol_list)

def get_molecules(dataframe):
    dataframe_reduced = dataframe
    dataframe_rejected = pd.DataFrame()
    nbrStruc = dataframe.shape[0]
    mol_list = []
    legends = []
    for i in range(nbrStruc):
        refcode = dataframe.at[i,"refcode_csd"]
        #print(refcode)       
        smile = dataframe.at[i,"smiles"]
        #print(i, refcode, "smile =", smile)
        # NOW TRY TO CONVERT SMILE INTO MOLECULE
        # THERE COULD BE SOME PROBLEMS DOING THIS CONVERSION
        # WE NEED TO HANDLE THE EXCEPTIONS

        if smile is None:
            print("\n\n", i, refcode)
            dataframe_rejected = pd.concat([dataframe_rejected, dataframe_reduced[dataframe_reduced.refcode_csd == refcode]])
            dataframe_reduced = dataframe_reduced[dataframe_reduced.refcode_csd != refcode]
            dataframe_reduced = dataframe_reduced.reset_index(drop=True)
            print("Smile code is not provided. Droping it from the dataframe")
        else:
            mol = Chem.MolFromSmiles(smile)
        
            if mol is None:
                print("\n\n", i, refcode)
                dataframe_rejected = pd.concat([dataframe_rejected, dataframe_reduced[dataframe_reduced.refcode_csd == refcode]])
                dataframe_reduced = dataframe_reduced[dataframe_reduced.refcode_csd != refcode]
                dataframe_reduced = dataframe_reduced.reset_index(drop=True)
                print("problem converting molecule. Droping it from the dataframe")

            else:    
                legends.append(refcode)
                #mol = Chem.AddHs(mol)
                mol_list.append(mol)
 
    dataframe_rejected.to_csv("../data/df_rejected.csv")      
    #Draw.MolsToGridImage(mol_list, molsPerRow=4, legends = legends)
    return(dataframe_reduced, mol_list, legends)

def get_molecular_features(dataframe, mol_list):
    df = dataframe
    for i in range(len(mol_list)):
        print("Getting molecular features for molecule: ", i)
        mol = mol_list[i]
        natoms = mol.GetNumAtoms()
        nbonds = mol.GetNumBonds()
        mw = Descriptors.ExactMolWt(mol)
        df.at[i,"NbrAtoms"] = natoms
        df.at[i,"NbrBonds"] = nbonds
        df.at[i,"mw"] = mw
        df.at[i,'HeavyAtomMolWt'] = Chem.Descriptors.HeavyAtomMolWt(mol)
        df.at[i,'NumValenceElectrons'] = Chem.Descriptors.NumValenceElectrons(mol)
        ''' # These four descriptors are producing the value of infinity for refcode_csd = YOLJUF (CCOP(=O)(Cc1ccc(cc1)NC(=S)NP(OC(C)C)(OC(C)C)[S])OCC\t\n)
        df.at[i,'MaxAbsPartialCharge'] = Chem.Descriptors.MaxAbsPartialCharge(mol)
        df.at[i,'MaxPartialCharge'] = Chem.Descriptors.MaxPartialCharge(mol)
        df.at[i,'MinAbsPartialCharge'] = Chem.Descriptors.MinAbsPartialCharge(mol)
        df.at[i,'MinPartialCharge'] = Chem.Descriptors.MinPartialCharge(mol)
        '''
        df.at[i,'FpDensityMorgan1'] = Chem.Descriptors.FpDensityMorgan1(mol)
        df.at[i,'FpDensityMorgan2'] = Chem.Descriptors.FpDensityMorgan2(mol)
        df.at[i,'FpDensityMorgan3'] = Chem.Descriptors.FpDensityMorgan3(mol)
        
        #print(natoms, nbonds)
        
        # Now get some specific features
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        feats = factory.GetFeaturesForMol(mol)
        #df["Acceptor"] = 0
        #df["Aromatic"] = 0
        #df["Hydrophobe"] = 0
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
                pass#print(feats[j].GetFamily())
                        
        df.at[i,"Acceptor"] = nbrAcceptor
        df.at[i,"Donor"] = nbrDonor
        df.at[i,"Hydrophobe"] = nbrHydrophobe
        df.at[i,"LumpedHydrophobe"] = nbrLumpedHydrophobe
        df.at[i,"PosIonizable"] = nbrPosIonizable
        df.at[i,"NegIonizable"] = nbrNegIonizable
        
        # We can also get some more molecular features using rdMolDescriptors
        
        df.at[i,"NumRotatableBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        df.at[i,"CalcChi0n"] = rdMolDescriptors.CalcChi0n(mol)
        df.at[i,"CalcChi0v"] = rdMolDescriptors.CalcChi0v(mol)
        df.at[i,"CalcChi1n"] = rdMolDescriptors.CalcChi1n(mol)
        df.at[i,"CalcChi1v"] = rdMolDescriptors.CalcChi1v(mol)
        df.at[i,"CalcChi2n"] = rdMolDescriptors.CalcChi2n(mol)
        df.at[i,"CalcChi2v"] = rdMolDescriptors.CalcChi2v(mol)
        df.at[i,"CalcChi3n"] = rdMolDescriptors.CalcChi3n(mol)
        #df.at[i,"CalcChi3v"] = rdMolDescriptors.CalcChi3v(mol)
        df.at[i,"CalcChi4n"] = rdMolDescriptors.CalcChi4n(mol)
        df.at[i,"CalcChi4v"] = rdMolDescriptors.CalcChi4v(mol)
        df.at[i,"CalcFractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
        df.at[i,"CalcHallKierAlpha"] = rdMolDescriptors.CalcHallKierAlpha(mol)
        df.at[i,"CalcKappa1"] = rdMolDescriptors.CalcKappa1(mol)
        df.at[i,"CalcKappa2"] = rdMolDescriptors.CalcKappa2(mol)
        #df.at[i,"CalcKappa3"] = rdMolDescriptors.CalcKappa3(mol)
        df.at[i,"CalcLabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
        df.at[i,"CalcNumAliphaticCarbocycles"] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
        df.at[i,"CalcNumAliphaticHeterocycles"] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
        df.at[i,"CalcNumAliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        df.at[i,"CalcNumAmideBonds"] = rdMolDescriptors.CalcNumAmideBonds(mol)
        df.at[i,"CalcNumAromaticCarbocycles"] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
        df.at[i,"CalcNumAromaticHeterocycles"] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        df.at[i,"CalcNumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        df.at[i,"CalcNumBridgeheadAtoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        df.at[i,"CalcNumHBA"] = rdMolDescriptors.CalcNumHBA(mol)
        df.at[i,"CalcNumHBD"] = rdMolDescriptors.CalcNumHBD(mol)
        df.at[i,"CalcNumHeteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        df.at[i,"CalcNumHeterocycles"] = rdMolDescriptors.CalcNumHeterocycles(mol)
        df.at[i,"CalcNumLipinskiHBA"] = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        df.at[i,"CalcNumLipinskiHBD"] = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        df.at[i,"CalcNumRings"] = rdMolDescriptors.CalcNumRings(mol)
        df.at[i,"CalcNumSaturatedCarbocycles"] = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
        df.at[i,"CalcNumSaturatedHeterocycles"] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
        df.at[i,"CalcNumSaturatedRings"] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        df.at[i,"CalcNumSpiroAtoms"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        df.at[i,"CalcTPSA"] = rdMolDescriptors.CalcTPSA(mol)
    return(df)
def createXYZ_from_SMILES(df, mol_list):
    new_mol_list = []
    print("Creating XYZ coordinates from SMILES")
    df = df.astype('object')
    df['xyz'] = ''
    for i, row in df.iterrows():
        #print(i, "Creating XYZ coordinates for mol with SMILES code = ", row['smiles'])
        mol = Chem.MolFromSmiles(row['smiles'])
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol)
        xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
        if (xyz is ''):
            print(i, "Unable to create XYZ coordinates for", row['smiles'], "Droping it from the dataframe")        
            df.at[i,"xyz"] = 'drop'
        else:
            new_mol_list.append(mol_list[i])       
            df.at[i,"xyz"] = xyz
    df = df[df["xyz"] != 'drop']
    df = df.reset_index(drop=True)
    return(df, new_mol_list)


    
