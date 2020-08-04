import numpy as np
import pandas as pd
import sys
import gnn_tools as gnn
import pickle
from dscribe.descriptors import MBTR
import subprocess
import glob, os

batch_size = 8000
df_reduced = pd.read_pickle("../data/df_reduced2.pic")
df = df_reduced[['xyz','smiles']]
del df_reduced
bf = gnn.Build_features(df,1, 1, 20, 5)  # df, addH, XYZ, nbr_Gaussian, NB_cutoff
######################### SETUP MBTR
atomicNbrs, elements, values = bf.getAtomicDist(df)
df = df.drop(['smiles'], axis=1)
# Setup MBTR
mbtr_constructor = MBTR(
    species=elements,
    k1={
        "geometry": {"function": "atomic_number"},
        "grid": {"min": 0, "max": 17, "n": 100, "sigma": 0.01},
    },
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.01},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    k3={
        "geometry": {"function": "cosine"},
        "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.01},
        "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization="n_atoms",
)
######################### GET MBTR features

for f in glob.glob("../data/mbtr*.pic"):
    os.remove(f)

counter = 0
for i in range(0,len(df), batch_size):
    counter += 1
    df_temp = df[i:i+batch_size]
    print("Batch =",counter)
    #print(df_temp)
    mbtr = bf.get_MBTR_feature(df_temp, mbtr_constructor)
    mbtr.to_pickle("../data/mbtr%s.pic"%counter)

######################### READ DATASET AND SAVE IT
    
dataset = gnn.Make_graph_normal(root="../data/processed_normal")
with open("../data/processed_normal/dataset.pic", 'wb') as filehandle:
    pickle.dump(dataset, filehandle, protocol=4)





 

