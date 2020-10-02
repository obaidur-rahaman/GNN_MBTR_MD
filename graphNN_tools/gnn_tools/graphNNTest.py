import torch
import numpy as np
import pandas as pd
import itertools
import math
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import NNConv, GATConv, Set2Set
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, GRU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops,add_remaining_self_loops
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from inits import reset, uniform
    
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d as BN
from torch.nn import LayerNorm as LN
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

################

def evaluate_GNN(getloss1, verbose1, target_term1, dataset, split_size, batch_size, GNN1, MD1, MBTR1, testData_has_target_values1):  
    #print(getloss1, verbose1, target_term1, dataset, split_size, num_epochs1, learningRate, batch_size, p1a, p2a, p_mda, p_mbtra, GNN1, MD1, MBTR1)
    print("Using batch_size: ", batch_size)
    global scale_target, getloss, p1, p2, p_md, p_mbtr, verbose, lr, nbrGlobalFeatures, nbrMBTRFeatures, targetNbr, numTarget, GNN, MD, MBTR, batch_size1, target_term, sel_target, y_mean, y_std
    global testData_has_target_values  
    sel_target = 1
    scale_target = 1
    getloss = getloss1
    verbose = verbose1
    nbrSample = len(dataset) # features.shape[0]
    numTarget = len(dataset[0].y)
    GNN = GNN1
    MD = MD1
    MBTR = MBTR1
    batch_size1 = batch_size
    target_term = target_term1
    testData_has_target_values = testData_has_target_values1
    # Determine which target number
    
    file1 = open('../data/target_terms_all.txt', 'r') 
    Lines = file1.readlines() 
    for i, target_term1 in enumerate(Lines): 
        #print("target_term1 =", target_term1, "target term =", target_term)
        target_term1 = target_term1.strip()
        if (target_term == target_term1):            
            targetNbr = i    
    if (1 == verbose):
        print("target =", target_term)
        print("Total number of datapoints =", nbrSample)
   
    nbrGlobalFeatures = 0
    nbrMBTRFeatures = 0
    # Check if the dataset contains some global features
    if (1 == MD):
        try: 
            u = dataset[0].u
            nbrGlobalFeatures = u.size()[0]
            if (1 == verbose):
                print("Number of global features assigned =", nbrGlobalFeatures)
        except:
            pass
    # Check if the dataset contains some MBTR features
    if (1 == MBTR):
        try: 
            mbtr = dataset[0].mbtr
            nbrMBTRFeatures = mbtr.size()[0]
            if (1 == verbose):
                print("Number of MBTR features assigned =", nbrMBTRFeatures)
        except:
            pass
    
    # Now find the mean and std of target
    if (1 == scale_target):
        y_values = []
        for i in range(len(dataset)):
            y_values.append(dataset[i].y[targetNbr])
        y_values = np.array(y_values)
        y_mean = y_values.mean()
        y_std = y_values.std()   

    test_dataset = dataset    
            
    #Now setup the dataloader
    
    from torch_geometric.data import DataLoader
    batch_size= batch_size
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=4)

    ############ Model
    global nbr_node_features, nbr_edge_features, in_channel, out_channel 
    nbr_node_features = dataset[0].x.size()[1]
    try:
        nbr_edge_features = dataset[0].edge_attr.size()[1]
    except:
        nbr_edge_features = 1
        
    in_channel = nbr_node_features
    out_channel = 1    
   
    # our model
    global device, model

    with open("../data_train_test_seprately/model.pic", 'rb') as f:
        model = pickle.load(f)     
  
    testData = get_results(test_loader)
    return(testData)    

def get_results(test_loader):  
    # for test 
   
    preds = []
    target = []     
    for data in test_loader:
        device = torch.device('cuda')
        data = data.to(device)
        output = model(data).detach()
        output = output.cpu()  
        if (1 == testData_has_target_values):
            label = data.y
            if (1 == sel_target):
                label = label.reshape(-1,numTarget)
                label = label[:,targetNbr]
            label = label.cpu()
            target.append(label)
        if (1 == scale_target):
            preds.append(output * y_std + y_mean)
        else:
            preds.append(output)           
    preds = torch.cat(preds,0)
    if (1 == testData_has_target_values):
        target = torch.cat(target,0)
        #print("Datapoints in the validation set =", len(preds))    
        testData = pd.DataFrame({'Target': target, 'Preds': preds})
    else:
        testData = pd.DataFrame({'Preds': preds})
    return(testData)

