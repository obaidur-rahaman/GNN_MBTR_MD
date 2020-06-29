import pickle
import matplotlib.pyplot as plt
import gnn_tools as gnn
from statistics import median
import numpy as np
import sys

def run_hyper_parallel(i, j, p, target_term, split_size, nbrEpochs, nbrGrid, param_best, GNN, MD, MBTR, show):
    #print(i, j, p, target_term, dataset, split_size, nbrEpochs, nbrGrid, param_best, MD, show)
    with open("./hyper/dataset.pic", 'rb') as f:
        dataset = pickle.load(f)
    trainLoss, valLoss = gnn.fit_GNN(1, 0, target_term, dataset, split_size, nbrEpochs, *param_best, GNN, MD, MBTR)
    #print("TrainLoss_trial =", trainLoss, "ValidationLoss_trial =", valLoss)

    print("param", i, "=", p, "Training loss =", round(trainLoss,6), "Validation loss =", round(valLoss,6))
        
    with open("./hyper/train_loss_%s.pic"%j, 'wb') as filehandle:
            pickle.dump(trainLoss, filehandle)
    with open("./hyper/val_loss_%s.pic"%j, 'wb') as filehandle:
            pickle.dump(valLoss, filehandle)

j = sys.argv[1]
#print("Inside hyperParOpt file")
with open("./hyper/Hyper_arguments_list_%s.pic"%j, 'rb') as f:
    argument_list = pickle.load(f)

run_hyper_parallel(*argument_list)

