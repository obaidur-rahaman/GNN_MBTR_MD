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
end = 4060

######################### PREPROCESS AND GENERATE DATASET

#df = pd.read_csv("../data_train_test_seprately/df_test.csv")
#df = df.drop(['xyz'], axis=1)
#df = df[["smiles","refcode_csd","homo"]]
#'''
df = pd.read_csv("../data_train_test_seprately/test_wt.csv")
df = df.reset_index()
df["refcode_csd"] = df["SMILES"]
df["smiles"] = df["SMILES"]
df["water"]= df["ln(gamma)_water"]
df = df[["smiles","refcode_csd","water"]]
#'''
df_reduced, mol_list = gnn.preprocessData(df, end, 1) #df, end, shuffle
# In case XYZ coordinates are not available, they can be generated
if 'xyz' not in df:
    print("XYZ coordinates do not exist. Creating from SMILES.")
    df_reduced, mol_list = gnn.createXYZ_from_SMILES(df_reduced, mol_list)

df_reduced.to_pickle("../data_train_test_seprately/df_reduced1.pic")
with open("../data_train_test_seprately/mol_list", "wb") as fp:
    pickle.dump(mol_list, fp)


df_reduced = pd.read_pickle("../data_train_test_seprately/df_reduced1.pic")
with open("../data_train_test_seprately/mol_list", 'rb') as f:
    mol_list = pickle.load(f)

subprocess.run(["rm", "-r", "../data_train_test_seprately/processed"])
bf = gnn.Build_features_train_test_separate(df_reduced, 1, 1, 20, 5, 0)  # df, addH, XYZ, nbr_Gaussian, NB_cutoff, called first time
df_reduced = bf.get_all_features(df_reduced, mol_list)

# Normalize features
# First open the mean and std that were calculated from the training set

with open("../data_train_test_seprately/mean_values_x.pic", 'rb') as f:
    mean_values_x = pickle.load(f)
with open("../data_train_test_seprately/mean_values_edge.pic", 'rb') as f:
    mean_values_edge = pickle.load(f)
with open("../data_train_test_seprately/std_values_x.pic", 'rb') as f:
    std_values_x = pickle.load(f)
with open("../data_train_test_seprately/std_values_edge.pic", 'rb') as f:
    std_values_edge = pickle.load(f)
with open("../data_train_test_seprately/min_max_scaler.pic", 'rb') as f:
    min_max_scaler = pickle.load(f)

print(mean_values_x)
df_reduced  = bf.normalize_featuresTest('x', mean_values_x, std_values_x, min_max_scaler)  
df_reduced  = bf.normalize_featuresTest('edge_attr', mean_values_edge, std_values_edge, min_max_scaler)
df_reduced  = bf.normalize_featuresTest('u', mean_values_x, std_values_x, min_max_scaler)

df_reduced.to_pickle("../data/df_reduced2.pic")







 

