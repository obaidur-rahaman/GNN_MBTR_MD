import numpy as np
import pandas as pd
import sys
import gnn_tools as gnn
from rdkit.Chem import Draw
import subprocess
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import time
from sklearn.metrics import mean_absolute_error
######################### INITIALIZE SOME VALUES
end = 1000 #133885
show_plots = 0
GNN = 1
MD = 1
MBTR = 1
batch_size = 64
testData_has_target_values = 0
######################### READ DATASET
if (0 == MBTR):
    dataset = gnn.Make_graph_inMemory(root="../data_train_test_seprately/")
    dataset = dataset[:end]
else:
    with open("../data_train_test_seprately/processed_normal/dataset.pic", 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    dataset, temp = torch.utils.data.random_split(dataset,(end,(len(dataset)-end)))
######################### SET UP INITIAL PARAMETERS
for target_term in ['homo']: #['homo', 'lumo']:
    print(target_term)       
    
    ######################### FINAL OPTIMIZATION
    print("########## ",target_term," GNN =", GNN, "MD = ",MD, "MBTR =", MBTR, "#############")
    print("Molecular Descriptor used =", MD)
    # getloss, verbose, target_term, dataset, split_size, num_epochs, lr, batch_size,  p1, p2, numLayer, numFinalFeature, GNN, MD, MBTI
    testData = gnn.evaluate_GNN(0, 1, target_term, dataset, 0.95, batch_size, GNN, MD, MBTR, testData_has_target_values)
    testData.to_csv("../results/%s/testFinal_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))

    if (testData_has_target_values == 1):  
        trainData = pd.read_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
        gnn.plot_results(trainData, testData, target_term, show = show_plots)
        
        # Now store the final result
        MAE = round(mean_absolute_error(testData["Preds"].to_numpy(), testData["Target"].to_numpy()), 4)
        RMSE = round(np.sqrt(mean_squared_error(testData["Preds"].to_numpy(), testData["Target"].to_numpy())), 4)
        with open("../results/all_results.txt", "a") as file_object:
            file_object.write("%s = %s (%s)   (GNN = %s  MD = %s  MBTR = %s)\n" % (target_term, MAE, RMSE, GNN, MD, MBTR))


