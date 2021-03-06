import pandas as pd
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.analysis.graphs import StructureGraph
import pymatgen.analysis.local_env as locenv
import pymatgen.io.xyz as pyxyz
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import os
from rdkit.Chem import ChemicalFeatures, rdMolChemicalFeatures
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory
from rdkit import RDConfig
from sklearn.utils import shuffle
import pickle 

def preprocessData(df, end, target_term): 
         
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df[:end]    
    df_reduced, mol_list, legends = get_molecules(df)
    print("Size of original dataframe =",len(df))
    print("Size of reduced dataframe =", len(df_reduced), "(excluding the cases that could not be processed)")
    
    df_reduced.to_pickle("../data/df_reduced.pic")
    #df_reduced.to_pickle("../data/df_reduced_%s.pic"%target_term)
    with open("../data/mol_list", "wb") as fp:
        pickle.dump(mol_list, fp)
    with open("../data/target_term.txt", "w") as f:
        f.write(target_term)
    return(df_reduced, mol_list)

def get_molecules(dataframe):
    dataframe_reduced = dataframe
    nbrStruc = dataframe.shape[0]
    mol_list = []
    legends = []
    for i in range(nbrStruc):
        filename = dataframe.at[i,"refcode_csd"]
        #print(filename)       
        smile = dataframe.at[i,"canonical_smiles"]
        #print(i, filename, "smile =", smile)
        # NOW TRY TO CONVERT SMILE INTO MOLECULE
        # THERE COULD BE SOME PROBLEMS DOING THIS CONVERSION
        # WE NEED TO HANDLE THE EXCEPTIONS
        try:
            smile = smile.split(".")[0]
        except:
            print(i, filename)
            print("smile =", smile)
            dataframe_reduced = dataframe_reduced[dataframe_reduced.refcode_csd != filename]
            dataframe_reduced = dataframe_reduced.reset_index(drop=True)
            print("Smile code is nonsensical. Droping it from the dataframe")
        #print("smile =", smile)
        else:
            mol = Chem.MolFromSmiles(smile)
        
            if mol is None:
                print(i, filename)
                print("smile =", smile)
                # WHEN NO MOLECULE CAN BE GENERATED BY THE SMILE CODE PROVIDED, WE CAN TRY TO 
                # GET IT FROM THE CSD USING THE REFCODE
                
                # FIRST WRITE THE REFCODE
                    with open("refcode.txt", "w") as f:
                        f.write(filname)
                '''
                dataframe_reduced = dataframe_reduced[dataframe_reduced.refcode_csd != filename]
                dataframe_reduced = dataframe_reduced.reset_index(drop=True)
                '''
                print("problem converting molecule. Droping it from the dataframe")
            else:
                legends.append(filename)
                mol = Chem.AddHs(mol)
                natom_smile = mol.GetNumAtoms() 
                #print("(number of atoms) =",natom_smile)
                mol_list.append(mol)
    #Draw.MolsToGridImage(mol_list, molsPerRow=4, legends = legends)
    return(dataframe_reduced, mol_list, legends)

def get_mol_pymatgen(dataframe, nbrStruc):
    
    for i in range(nbrStruc):
        filename = dataframe.iloc[i,0]
        if (filename != "QAPHUK"):
            print(filename)
            parser = CifParser("../data/cif/{}.cif".format(filename))
            structure = parser.get_structures()[0]
            sg = StructureGraph.with_local_env_strategy(structure, locenv.JmolNN())
            my_molecules = sg.get_subgraphs_as_molecules()
            #molecule = MoleculeGraph.with_local_env_strategy(my_molecules,locenv.JmolNN())
            #print(molecule)
            #crystool.view(my_molecules)
            my_molecules = pyxyz.XYZ(my_molecules)
            #print(my_molecules)
            pyxyz.XYZ.write_file(my_molecules,"../data/xyz_pymatgen/{}.xyz".format(filename))
            
def get_smile(dataframe):
    nbrStruc = dataframe.shape[0]
    for i in range(nbrStruc):
        filename = dataframe.iloc[i,0]
        #print(filename, "========================")
        f=open("../data/can/{}.can".format(filename), "r")
        smile = f.read().split()
        smile = smile[0]
        # But let's first make sure there is only one molecule present    
        smile = smile.split(".")[0]
        
        dataframe.at[i,"smile"] = smile
        
        # Now read the number of atoms derived from openbabel        
        with open("../data/xyz_openbabel/{}.xyz".format(filename)) as f:
            natom_obabel = f.readline().rstrip()
            #print("natom_obabel =", natom_obabel)
            dataframe.at[i,"Natom_obabel"] = natom_obabel
            
        # Now read the number of atoms derived from pymatgen        
        with open("../data/xyz_pymatgen/{}.xyz".format(filename)) as f:
            natom_pymatgen = f.readline().rstrip()
            #print("natom_pymatgen =", natom_pymatgen)
            dataframe.at[i,"Natom_pymatgen"] = natom_pymatgen       
            
        if (natom_obabel != natom_pymatgen):
            dataframe.at[i,"Match"] = "No"
            
            # In case openbabel fails to create the molecule (splits it into two)
            # Use the pymatgen generated xyz to create the smile code            
            subprocess.run(["obabel", "../data/xyz_pymatgen/{}.xyz".format(filename), "-O", "../data/xyz_pymatgen/temp.can"])
            f=open("../data/xyz_pymatgen/temp.can", "r")
            smile = f.read().split()
            smile = smile[0]
            # But let's first make sure there is only one molecule present    
            smile = smile.split(".")[0]
            dataframe.at[i,"smile"] = smile
            print(filename, "========================")
            print("natom_obabel =", natom_obabel)
            print("natom_pymatgen =", natom_pymatgen)
            print("Atoms did not match between openbabel and pymatgen")
            print("Canonical SMILE code =",smile)
        else:
            dataframe.at[i,"Match"] = "-"
            
    return(dataframe)

def get_smile_from_CSD(dataframe):
    nbrStruc = dataframe.shape[0]
    from ccdc import io
    csd_reader = io.EntryReader('CSD')
    for i in range(nbrStruc):
        filename = dataframe.iloc[i,0]
        print(filename)
        mol = csd_reader.molecule(filename)
        smile= mol.smiles
        print(smile)
        dataframe.at[i,"smile"] = smile
    return(dataframe)

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
        df.at[i,"CalcChi3v"] = rdMolDescriptors.CalcChi3v(mol)
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
    
