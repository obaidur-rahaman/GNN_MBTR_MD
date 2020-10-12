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

def fit_GNN(getloss1, verbose1, target_term1, dataset, split_size, num_epochs1, learningRate, batch_size, p1a, p2a, p_mda, p_mbtra, GNN1, MD1, MBTR1):  
    #print(getloss1, verbose1, target_term1, dataset, split_size, num_epochs1, learningRate, batch_size, p1a, p2a, p_mda, p_mbtra, GNN1, MD1, MBTR1)
    print("Using hyperparameters: ", learningRate, batch_size, p1a, p2a, p_mda, p_mbtra)
    global scale_target, getloss, p1, p2, p_md, p_mbtr, verbose, lr, nbrGlobalFeatures, nbrMBTRFeatures, targetNbr, numTarget, GNN, MD, MBTR, batch_size1, target_term, sel_target, y_mean, y_std  
    sel_target = 1
    scale_target = 1
    getloss = getloss1
    lr = learningRate
    p1 = p1a
    p2 = p2a
    p_md = p_mda
    p_mbtr = p_mbtra
    verbose = verbose1
    nbrSample = len(dataset) # features.shape[0]
    numTarget = len(dataset[0].y)
    GNN = GNN1
    MD = MD1
    MBTR = MBTR1
    batch_size1 = batch_size
    target_term = target_term1
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
    
    l1 = int(round(nbrSample*split_size))
    l2 = int(round(nbrSample*(0.5*(1 + split_size))))
    epoch_early_stop = 70   
    
    if (0 == getloss):
        #dataset = dataset.shuffle()
        valFreq1a = 9
        valFreq1b = 10
        valFreq2a = epoch_early_stop
        valFreq2b = 10
   
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

    if (1 == MBTR):
        train_dataset = torch.utils.data.Subset(dataset, list(range(0,l1)))
        val_dataset = torch.utils.data.Subset(dataset, list(range(l1, l2)))
        test_dataset = torch.utils.data.Subset(dataset, list(range(l2, len(dataset))))
        #train_dataset,test_val_dataset = torch.utils.data.random_split(dataset,(l1,(nbrSample-l1)))
        #val_dataset, test_dataset = torch.utils.data.random_split(test_val_dataset,((l2-l1),(nbrSample-l2)))
    else:
        train_dataset = dataset[:l1]
        val_dataset = dataset[l1:l2]
        test_dataset = dataset[l2:]

    print("TrainsetSize = ", len(train_dataset), "ValidationsetSize = ", len(val_dataset), "TestsetSize = ", len(test_dataset))    
            
    #Now setup the dataloader
    
    from torch_geometric.data import DataLoader
    batch_size= batch_size
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=4)
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
    device = torch.device('cuda')
    if (1 == getloss):
        torch.manual_seed(836)
    model = Net().to(device)     
        
    # Construct our loss function and an Optimizer. 
    global criterion, optimizer
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

    global num_epochs
    num_epochs = num_epochs1
       
    # Now train
    torch.backends.cudnn.benchmark = True
    df_loss = pd.DataFrame(columns = ["epoch", "train_loss", "validation_loss"])
    val_err_3rdlast = 0
    val_err_2ndlast = 0
    val_err_last = 0
    val_acc = 0
    counter = 0
    flag = 0
    best_val_error = None
    for epoch in range(num_epochs):  
        if 1 == flag:
            break               
        df_loss.loc[epoch,"epoch"] = epoch
        loss = train(epoch, train_loader, train_dataset)
        val_error = test(val_loader)
        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            #test_error = test(test_loader)
            best_val_error = val_error
        if (1 == verbose):
            if 1 == scale_target:
                print("epoch =", epoch, "  loss =",loss, "Validation MAE =",val_error * y_std)
            else:
                print("epoch =", epoch, "  loss =",loss, "Validation MAE =",val_error)
        df_loss.loc[epoch,"train_loss"] = loss   

        if (0 == getloss): 
        # Regularly get the validation accuracy
        # Check less frequently when epoch < epoch_early_stop and more frequently later
            if (epoch < epoch_early_stop):
                valFreqA = valFreq1a
                valFreqB = valFreq1b
            else:
                valFreqA = valFreq2a
                valFreqB = valFreq2b

            for ep in range(valFreqA, num_epochs, valFreqB):             
                if (epoch == ep):  
                    counter = counter + 1
                    val_acc = evaluate(val_loader,val_dataset)  
                    df_loss.loc[epoch,"validation_loss"] = val_acc
                    if (1 == verbose):    
                        print('Epoch: {:03d}, Train error: {:.5f}, Val error: {:.5f}'.format(epoch, loss, val_acc))
                        
                    # Store the loss and estimate how learning is saturated with number of epochs                    
                    df_loss.to_csv("../results/%s/loss_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR), index = False)                       
                    # Also store the test results
                    trainData, testData = get_results(train_loader, test_loader)
                    trainData.to_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
                    testData.to_csv("../results/%s/test_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
                    # EARLY STOPPING: Stop if the validation error starts to go up
                    #print(val_err_3rdlast, val_err_2ndlast, val_err_last, val_acc, "counter =",counter)
                    '''
                    if (epoch > epoch_early_stop):
                        if (val_acc > val_err_last) and (val_err_last > val_err_2ndlast) and (val_err_2ndlast > val_err_3rdlast):
                            print("Stopped training because validation error is gradually going up!!!")
                            flag = 1
                            break                                                  
                    val_err_3rdlast = val_err_2ndlast
                    val_err_2ndlast = val_err_last
                    val_err_last = val_acc
                    '''
                    break       
        #print(df_loss) 
    with open('../data_train_test_seprately/model.pic', 'wb') as b:
        pickle.dump(model,b)       
    if (0 == getloss):  
        trainData, testData = get_results(train_loader, test_loader)
        return(trainData, testData)
    else:
        val_acc = evaluate(val_loader,val_dataset)
        return(loss, val_acc)    

#Now setup the message passing

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.GNN = GNN
        self.MD = MD
        self.MBTR = MBTR
        self.nbrGlobalFeatures = nbrGlobalFeatures
        self.nbrMBTRFeatures = nbrMBTRFeatures
        '''
        nn (torch.nn.Module) – A neural network ℎ𝚯 that maps edge features 
        edge_attr of shape [-1, num_edge_features] to 
        shape [-1, in_channels * out_channels], e.g., 
        defined by torch.nn.Sequential.
        '''   
        totNbrFeatures = 0  
        if (1 == self.GNN): 
            totNbrFeatures += p2*2
            self.lin0 = torch.nn.Linear(nbr_node_features, p2, bias = False)
            self.BN0 = BN(round(p2))
            nn = Seq(Linear(nbr_edge_features, p1, bias = False), BN(p1), LeakyReLU(), Linear(p1, p2 * p2, bias = False), BN(p2 * p2))
            self.conv = NNConv(p2, p2,nn, aggr='mean')

            self.set2set = Set2Set(p2, processing_steps=3)    
            #print("nbrGlobalFeatures =", nbrGlobalFeatures,"nbrMBTRFeatures =", nbrMBTRFeatures)        
            self.gru = GRU(p2, p2)       
        if (1 == self.MD):
            totNbrFeatures += p_md
            self.lin_MD = torch.nn.Linear(self.nbrGlobalFeatures, 4 * p_md)
            self.lin_MD2 = torch.nn.Linear(4 * p_md, 2 * p_md)
            self.lin_MD3 = torch.nn.Linear(2 * p_md, p_md)
        if (1 == self.MBTR):
            totNbrFeatures += p_mbtr
            self.lin_MBTR = torch.nn.Linear(self.nbrMBTRFeatures, 4 * p_mbtr, bias = False)
            self.BN_MBTR1 = BN(4 * p_mbtr)
            self.lin_MBTR2 = torch.nn.Linear(4 * p_mbtr, 2 * p_mbtr, bias = False)
            self.BN_MBTR2 = BN(2 * p_mbtr) 
            self.lin_MBTR3 = torch.nn.Linear(2 * p_mbtr, p_mbtr, bias = False)
            self.BN_MBTR3 = BN(p_mbtr)

        print("totNbrFeatures", totNbrFeatures) 
        self.lin1 = torch.nn.Linear(totNbrFeatures, round(totNbrFeatures/2))  
        self.lin2 = torch.nn.Linear(round(totNbrFeatures/2), round(totNbrFeatures/4))
        self.lin_final = torch.nn.Linear(round(totNbrFeatures/4), 1)
        
    def forward(self, data):
        y = None
        if (1 == self.GNN):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            out = F.leaky_relu(self.BN0(self.lin0(x)))
            h = out.unsqueeze(0)

            for i in range(3):
                m = F.leaky_relu(self.conv(out, edge_index, edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)

            y_gnn = self.set2set(out, batch)  
            y = y_gnn
        # Check if there are any global features         
        if (1 == self.MD):
            y_md = data.u
            y_md = y_md.reshape(-1, self.nbrGlobalFeatures)
            y_md = F.leaky_relu(self.lin_MD(y_md))
            y_md = F.leaky_relu(self.lin_MD2(y_md))
            y_md = F.leaky_relu(self.lin_MD3(y_md))
            if y is None:
                y = y_md               
            else:
                y = torch.cat((y, y_md), 1)
        #print("1. size of y =", y.size())
        # Check if there are any MBTR features         
        if (1 == self.MBTR):
            y_mbtr = data.mbtr
            y_mbtr[y_mbtr != y_mbtr] = 0
            y_mbtr = y_mbtr.reshape(-1, self.nbrMBTRFeatures)
            y_mbtr = F.leaky_relu(self.BN_MBTR1(self.lin_MBTR(y_mbtr)))
            y_mbtr = F.leaky_relu(self.BN_MBTR2(self.lin_MBTR2(y_mbtr)))
            y_mbtr = F.leaky_relu(self.BN_MBTR3(self.lin_MBTR3(y_mbtr)))
            if y is None:
                y = y_mbtr
            else:
                y = torch.cat((y, y_mbtr), 1)
        y = F.dropout(y, p = 0.5, training=self.training)
        y = F.leaky_relu(self.lin1(y)) 
        y = F.leaky_relu(self.lin2(y))
        y = self.lin_final(y)
        y = y.squeeze(-1)
        return y   

# Training loop

def train(epoch, train_loader, train_dataset):
    model.train()
    loss_all = 0
    for data in train_loader:
        #print("in for loop of train_loader")
        data = data.to(device)
        #print("train definition, edge_attr =",data.edge_attr.size())
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        if (1 == sel_target):
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
        if (1 == scale_target):
            label = (label - y_mean) / y_std
        loss = criterion(output, label)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss_all += data.num_graphs * loss.item()                    
        optimizer.step()
        del loss, output, label
    return loss_all / len(train_dataset)

def test(loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            label = data.y.to(device)
            if (1 == sel_target):
                label = label.reshape(-1,numTarget)
                label = label[:,targetNbr]
            if (1 == scale_target):
                label = (label - y_mean) / y_std
            error += (model(data) - label).abs().sum().item()  # MAE
    return error / len(loader.dataset)

def evaluate(loader,whichDataset):   
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            label = data.y.to(device)
            if (1 == sel_target):
                label = label.reshape(-1,numTarget)
                label = label[:,targetNbr]
            if (1 == scale_target):
                label = (label - y_mean) / y_std
            loss = criterion(output, label)
            loss_all += data.num_graphs * loss.item()
    #print("loss =",loss_all / len(whichDataset))
    return loss_all / len(whichDataset)

def get_results(train_loader, test_loader): 

    # for training
    
    preds = []
    target = []     
    for data in train_loader:
        data = data.to(device)
        output = model(data).detach()
        #print("output =", output)
        output = output.cpu()  
        label = data.y.to(device)
        if (1 == sel_target):
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
        label = label.cpu()
        #print("label =", label)
        if (1 == scale_target):
            preds.append(output * y_std + y_mean)
        else:
            preds.append(output)
        target.append(label) 
    #print("preds =", len(preds), "target =", len(target))
    preds = torch.cat(preds,0)
    target = torch.cat(target,0)
    #print("Datapoints in the training set =", len(preds))      
    trainData = pd.DataFrame({'Target': target, 'Preds': preds})
    
    # for test 
    
    preds = []
    target = []     
    for data in test_loader:
        data = data.to(device)
        output = model(data).detach()
        output = output.cpu()  
        label = data.y.to(device)
        if (1 == sel_target):
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
        label = label.cpu()
        if (1 == scale_target):
            preds.append(output * y_std + y_mean)
        else:
            preds.append(output)
        target.append(label)   
    preds = torch.cat(preds,0)
    target = torch.cat(target,0)
    #print("Datapoints in the validation set =", len(preds))    
    testData = pd.DataFrame({'Target': target, 'Preds': preds})

    return(trainData, testData)

