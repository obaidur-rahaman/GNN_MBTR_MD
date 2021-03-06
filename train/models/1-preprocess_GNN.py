import numpy as np
import pandas as pd
import sys
import gnn_tools as gnn
from rdkit.Chem import Draw
import subprocess
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import time
import gc
import math
from rdkit.Chem import PandasTools
from rdkit import Chem
import re

######################### INITIALIZE SOME VALUES
end = 40 #133885

######################### PREPROCESS AND GENERATE DATASET

df = pd.read_csv("../data/df.csv")
df_reduced, mol_list = gnn.preprocessData(df, end, 1) # dataframe, end, shuffle
# In case XYZ coordinates are not available, they can be generated
if 'xyz' not in df:
    print("XYZ coordinates do not exist. Creating from SMILES.")
    df_reduced, mol_list = gnn.createXYZ_from_SMILES(df_reduced, mol_list)

df_reduced.to_pickle("../data/df_reduced1.pic")
with open("../data/mol_list", "wb") as fp:
    pickle.dump(mol_list, fp)


df_reduced = pd.read_pickle("../data/df_reduced1.pic")
with open("../data/mol_list", 'rb') as f:
    mol_list = pickle.load(f)

subprocess.run(["rm", "-r", "../data/processed"])
bf = gnn.Build_features(df_reduced, 1, 1, 20, 5)  # df, addH, XYZ, nbr_Gaussian, NB_cutoff
df_reduced = bf.get_all_features(df_reduced, mol_list)

# Normalize features

for i in ['x','edge_attr', 'u']:
    df_reduced = bf.normalize_features(i)  

df_reduced.to_pickle("../data/df_reduced2.pic")





 

