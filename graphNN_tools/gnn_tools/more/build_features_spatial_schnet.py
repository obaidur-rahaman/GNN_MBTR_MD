import numpy as np
import pandas as pd
import torch
import pickle
import os
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
from ase import Atoms
import re
from sklearn.utils import shuffle
from ase.io import read
from schnetpack import AtomsData

def get_all_features_spatial(df, end):
    df = shuffle(df)
    df = df.reset_index(drop=True)
    df = df[:end]
    xyz_all = ''
    for i, row in df.iterrows():
        #print(row['homo'])
        xyz = row['xyz']
        xyz_new = xyz.split("\n",2)[0] + '\n' + str(row['homo']) + '\n' + xyz.split("\n",2)[2]
        xyz_all = xyz_all + xyz_new


    with open("coord.xyz", "w") as xyz_file:
        xyz_file.write(xyz_all)

    atoms = read('coord.xyz', index=':10')
    property_list = []
    for at in atoms:
        # All properties need to be stored as numpy arrays.
        # Note: The shape for scalars should be (1,), not ()
        # Note: GPUs work best with float32 data
        homo = np.array([float(list(at.info.keys())[0])], dtype=np.float32)    
        property_list.append(
            {'homo': homo}
        )
        
    #print('Properties:', property_list)
    new_dataset = AtomsData('./new_dataset.db', available_properties=['homo'])
    new_dataset.add_systems(atoms, property_list)   

    '''
    print('Number of reference calculations:', len(new_dataset))
    print('Available properties:')

    for p in new_dataset.available_properties:
        print('-', p)
    print()    

    example = new_dataset[0]
    print('Properties of molecule with id 0:')

    for k, v in example.items():
        print('-', k, ':', v.shape)
    '''
    return(new_dataset)


