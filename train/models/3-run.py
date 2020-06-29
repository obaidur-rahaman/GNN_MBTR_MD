import numpy as np
import pandas as pd
import sys
import gnn_tools as gnn
from rdkit.Chem import Draw
import subprocess
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import time
from sklearn.metrics import mean_absolute_error
######################### INITIALIZE SOME VALUES
end = 2000 #133885
show_plots = 0
num_epochs = 150
GNN = 1
MD = 0
MBTR = 0
######################### READ DATASET
if (0 == MBTR):
    dataset = gnn.Make_graph_inMemory(root="../data/")
    dataset = dataset[:end]
else:
    with open("../data/processed_normal/dataset.pic", 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    dataset, temp = torch.utils.data.random_split(dataset,(end,(len(dataset)-end)))
######################### SET UP INITIAL PARAMETERS
for target_term in ['homo']: #['homo', 'lumo']:
    print(target_term)       
    if ("homo" == target_term): 
        hyper = 0
        param_best = [0.0027, 64, 79, 89, 50, 130]
        param_range = [[0.001, 0.01],[64, 64], [60, 100], [60, 100], [50, 50], [30,130]]
    elif ("lumo" == target_term):
        hyper = 0
        param_best = [0.007498,  32,  128,  64,  50,  100]
        param_range = [[0.001, 0.01],[64, 64], [67, 67], [38, 38], [50, 50], [40,150]]
    ######################### HYPER PARAMETER OPTIMIZATION
    if ( 1 == hyper):
        # Clean the hyper data
        subprocess.run(["rm", "-r", "hyper"]) 
        subprocess.run(["mkdir", "hyper"])
        with open("./hyper/dataset.pic", 'wb') as filehandle:
                    pickle.dump(dataset, filehandle, protocol=4)    
        # RANDOM SEARCH
        #hyper_batch_size, target_term, dataset1, split_size, parameter_ranges, nbrTrials, nbrEpochs, MD
        param_best, param_best_5 = gnn.fit_hyperParameters_random(1, target_term, 0.85, param_range, 10, 10, GNN, MD, MBTR)
    ######################### FINAL OPTIMIZATION
    print("########## ",target_term," GNN =", GNN, "MD = ",MD, "MBTR =", MBTR, "#############")
    print("Molecular Descriptor used =", MD)
    # getloss, verbose, target_term, dataset, split_size, num_epochs, lr, batch_size,  p1, p2, numLayer, numFinalFeature, GNN, MD, MBTI
    trainData, testData = gnn.fit_GNN(0, 1, target_term, dataset, 0.85, num_epochs, *param_best, GNN, MD, MBTR)
    trainData.to_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    testData.to_csv("../results/%s/test_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    trainData = pd.read_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    testData = pd.read_csv("../results/%s/test_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))        
    if (1 == show_plots):
        for i in range(num_epochs):
            gnn.plot_losses(target_term, GNN, MD, MBTR)
            time.sleep(30)
        gnn.plot_results(trainData, testData, target_term, show = show_plots)
    
    # Now store the final result
    MAE = mean_absolute_error(testData["Preds"].to_numpy(), testData["Target"].to_numpy())
    with open("../results/all_results.txt", "a") as file_object:
        file_object.write("%s = %s    (GNN = %s  MD = %s  MBTR = %s)\n" % (target_term, MAE, GNN, MD, MBTR))
