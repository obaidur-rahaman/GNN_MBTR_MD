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
end = 2000 #133885

######################### PREPROCESS AND GENERATE DATASET

df = pd.read_csv("../data_train_test_seprately/df_train.csv")
df_reduced, mol_list = gnn.preprocessData(df, end, 1) # dataframe, end, shuffle
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
bf = gnn.Build_features_train_test_separate(df_reduced, 1, 1, 20, 5, 1)  # df, addH, XYZ, nbr_Gaussian, NB_cutoff, called first time
df_reduced = bf.get_all_features(df_reduced, mol_list)

# Normalize features

df_reduced, mean_values_x, std_values_x = bf.normalize_features('x')  
df_reduced, mean_values_edge, std_values_edge = bf.normalize_features('edge_attr')
df_reduced, min_max_scaler = bf.normalize_features('u')

# Now store the mean and std values
with open('../data_train_test_seprately/mean_values_x.pic', 'wb') as b:
    pickle.dump(mean_values_x,b)
with open('../data_train_test_seprately/mean_values_edge.pic', 'wb') as b:
    pickle.dump(mean_values_edge,b)
with open('../data_train_test_seprately/std_values_x.pic', 'wb') as b:
    pickle.dump(std_values_x,b)
with open('../data_train_test_seprately/std_values_edge.pic', 'wb') as b:
    pickle.dump(std_values_edge,b)
with open('../data_train_test_seprately/min_max_scaler.pic', 'wb') as b:
    pickle.dump(min_max_scaler,b) 

df_reduced.to_pickle("../data/df_reduced2.pic")





 

