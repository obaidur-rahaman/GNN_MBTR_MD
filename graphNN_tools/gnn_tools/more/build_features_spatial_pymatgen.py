import numpy as np
import pandas as pd
import torch
import pickle
import os
import sys
sys.path
sys.path.append('/home/ge35gen/software/openbabel/build/bin/')
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase import Atoms
import re
from sklearn.utils import shuffle
from ase.io import read
from schnetpack import AtomsData
from pymatgen import Lattice, Structure, Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis import local_env
from pymatgen.analysis.local_env import NearNeighbors
from typing import Union
from pymatgen.io.xyz import XYZ
from openbabel import pybel

def get_all_features_spatial(df, end):
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df[:end]
    xyz_all = ''
    for i, row in df.iterrows():
        #print(row['homo'])
        xyz = row['xyz']
        mol = pb.readstring('smi', row['smile'])
        strategy = local_env.NearNeighbors()
        mol = Molecule.from_file("temp.xyz")
        Structure.from_file("temp.xyz")
        mol = Molecule.from_str(xyz, fmt = 'xyz') 
        molGraph = MoleculeGraph.with_local_env_strategy(mol,strategy)
        molGraph.get_connected_sites(2)

def get_pmg_mol_from_smiles(smiles: str) -> Molecule:
    """
    Get a pymatgen molecule from smiles representation
    Args:
        smiles: (str) smiles representation of molecule
    """
    b_mol = pb.readstring('smi', smiles)
    b_mol.make3D()
    b_mol = b_mol.OBMol
    p_mol = BabelMolAdaptor(b_mol).pymatgen_mol
    return p_mol