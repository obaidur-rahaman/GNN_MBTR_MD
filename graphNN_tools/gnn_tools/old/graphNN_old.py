import torch
import numpy as np
import pandas as pd
import itertools
import math
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops,add_remaining_self_loops
from torch.nn import Parameter
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from inits import reset, uniform
    
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
################

# We got the dataset. We can now split it into a training set, test set and validation set.

def fit_GNN(getloss1, verbose1, target_term1, dataset, split_size, num_epochs1, learningRate, batch_size, p1a, p2a, p_mda, p_mbtia, GNN1, MD1, MBTR1):  
    #print(getloss1, verbose1, target_term1, dataset, split_size, num_epochs1, learningRate, batch_size, p1a, p2a, p_mda, p_mbtia, GNN1, MD1, MBTR1)
    global getloss, p1, p2, p_md, p_mbti, verbose, lr, nbrGlobalFeatures, nbrMBTRFeatures, targetNbr, numTarget, GNN, MD, MBTR, batch_size1, target_term 
    getloss = getloss1
    lr = learningRate
    p1 = p1a
    p2 = p2a
    p_md = p_mda
    p_mbti = p_mbtia
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
    
    train_dataset,test_val_dataset = torch.utils.data.random_split(dataset,(l1,(nbrSample-l1)))
    val_dataset, test_dataset = torch.utils.data.random_split(test_val_dataset,((l2-l1),(nbrSample-l2)))
    #train_dataset = dataset[:l1]
    #val_dataset = dataset[l1:l2]
    #test_dataset = dataset[l2:]

    len(train_dataset), len(val_dataset)    
            
    #Now setup the dataloader
    
    from torch_geometric.data import DataLoader
    batch_size= batch_size
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers=4)

    ############ Model
    global nbr_node_features, nbr_edge_features, in_channel, out_channel 
    nbr_node_features = dataset[0].x.size()[1]
    nbr_edge_features = dataset[0].edge_attr.size()[1]
   
    in_channel = nbr_node_features
    out_channel = 1    
   
    # our model
    global device, model
    device = torch.device('cuda')
    if (1 == getloss):
        torch.manual_seed(836)
    model = Net().to(device)     
        
    
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    global criterion, optimizer
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    global num_epochs
    num_epochs = num_epochs1
       
    # Now train
    
    df_loss = pd.DataFrame(columns = ["epoch", "train_loss", "validation_loss"])
    val_err_3rdlast = 0
    val_err_2ndlast = 0
    val_err_last = 0
    val_acc = 0
    counter = 0
    flag = 0
    for epoch in range(num_epochs):  
        if 1 == flag:
            break               
        df_loss.loc[epoch,"epoch"] = epoch
        loss = train(epoch, train_loader, train_dataset)
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
                    '''
                    val_err_3rdlast = val_err_2ndlast
                    val_err_2ndlast = val_err_last
                    val_err_last = val_acc
                    break       
        #print(df_loss)        
    
    if (0 == getloss):  
        trainData, testData = get_results(train_loader, test_loader)
        return(trainData, testData)
    else:
        val_acc = evaluate(val_loader,val_dataset)
        return(loss, val_acc)
    
# Now define a function to create a pytorch geometric dataset

class lambda1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(lambda1, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']

    def download(self):
        pass
    
    def process(self):
        pass

        
#dataset = lambda1(root='./')   

        
#Now setup the message passing

class NNConv(MessagePassing):
    """The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)


    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        #print("within forward, pseudo =",pseudo.size())
        return self.propagate(edge_index, x=x, pseudo=pseudo)


    def message(self, x_j, pseudo):
        #print("within message, pseudo =",pseudo.size())
        #print("within message, in and out channels =",self.in_channels,"and", self.out_channels)
        #print("within message, nn(pseudo) =",self.nn(pseudo).size())
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        #print("within message, weight =",weight.size())
        #print("within message, x_j =",x_j.size())
        #print("within message, x_j.unsqueeze(1) =",x_j.unsqueeze(1).size())
        #print("within message, torch.matmul(x_j.unsqueeze(1), weight) =",torch.matmul(x_j.unsqueeze(1), weight).size())
        y = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        #print("within message, y =", y.size())
        return y

    def update(self, aggr_out, x):
        #print("within update, aggr_out init=", aggr_out.size())
        if self.root is not None:
            #print ("Within first if")
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            #print ("Within second if")
            aggr_out = aggr_out + self.bias
        #print("within update, aggr_out final=", aggr_out.size())
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        nn (torch.nn.Module) – A neural network ℎ𝚯 that maps edge features 
        edge_attr of shape [-1, num_edge_features] to 
        shape [-1, in_channels * out_channels], e.g., 
        defined by torch.nn.Sequential.
        '''      

        #nn1 = nn.Sequential(nn.Linear(nbr_edge_features, in_channel * out_channel))
        nn1 = nn.Sequential(nn.Linear(nbr_edge_features, p1), nn.ReLU(), nn.Linear(p1, in_channel* 3 * p2))
        self.conv1 = NNConv(in_channel, 3 * p2,nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(nbr_edge_features, p1), nn.ReLU(), nn.Linear(p1, 3 * p2* 2 * p2))
        self.conv2 = NNConv(3 * p2, 2 * p2,nn2, aggr='mean')

        nn3 = nn.Sequential(nn.Linear(nbr_edge_features, p1), nn.ReLU(), nn.Linear(p1, 2 * p2*p2))
        self.conv3 = NNConv(2 * p2, p2,nn3, aggr='mean')
        
        self.set2set = Set2Set(p2, processing_steps=2)    
        #print("nbrGlobalFeatures =", nbrGlobalFeatures,"nbrMBTRFeatures =", nbrMBTRFeatures)

        totNbrFeatures = 0
        if (1 == GNN):
            totNbrFeatures += p2*2
        if (1 == MD):
            totNbrFeatures += p_md
            self.lin_MD = torch.nn.Linear(nbrGlobalFeatures, 4 * p_md)
            self.lin_MD2 = torch.nn.Linear(4 * p_md, 2 * p_md)
            self.lin_MD3 = torch.nn.Linear(2 * p_md, p_md)
        if (1 == MBTR):
            totNbrFeatures += p_mbti
            self.lin_MBTR = torch.nn.Linear(nbrMBTRFeatures, 4 * p_mbti)
            self.lin_MBTR2 = torch.nn.Linear(4 * p_mbti, 2 * p_mbti)
            self.lin_MBTR3 = torch.nn.Linear(2 * p_mbti, p_mbti)

        print("totNbrFeatures", totNbrFeatures)
        self.lin1 = torch.nn.Linear(totNbrFeatures, round(totNbrFeatures/2))           
        self.lin2 = torch.nn.Linear(round(totNbrFeatures/2), round(totNbrFeatures/4))
        self.lin_final = torch.nn.Linear(round(totNbrFeatures/4), 1)
        
    def forward(self, data):
        y = None
        if (1 == GNN):
            y_gnn, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr           
            y_gnn = F.relu(self.conv1(y_gnn, edge_index, edge_attr))   
            y_gnn = F.relu(self.conv2(y_gnn, edge_index, edge_attr))   
            y_gnn = F.relu(self.conv3(y_gnn, edge_index, edge_attr))     
            batch = data.batch
            y_gnn = self.set2set(y_gnn, batch)  
            y = y_gnn
        # Check if there are any global features         
        if (1 == MD):
            y_md = data.u
            y_md = y_md.reshape(-1, nbrGlobalFeatures)
            y_md = F.relu(self.lin_MD(y_md))
            y_md = F.relu(self.lin_MD2(y_md))
            y_md = F.relu(self.lin_MD3(y_md))
            if y is None:
                y = y_md               
            else:
                y = torch.cat((y, y_md), 1)
        #print("1. size of y =", y.size())
        # Check if there are any MBTR features         
        if (1 == MBTR):
            y_mbtr = data.mbtr
            y_mbtr[y_mbtr != y_mbtr] = 0
            y_mbtr = y_mbtr.reshape(-1, nbrMBTRFeatures)
            y_mbtr = F.relu(self.lin_MBTR(y_mbtr))
            y_mbtr = F.relu(self.lin_MBTR2(y_mbtr))
            y_mbtr = F.relu(self.lin_MBTR3(y_mbtr))
            if y is None:
                y = y_mbtr
            else:
                y = torch.cat((y, y_mbtr), 1)

        y = F.relu(self.lin1(y)) 
        y = F.relu(self.lin2(y))   
        y = self.lin_final(y)
        y = y.squeeze(-1)
        return y   

# Training loop

def train(epoch, train_loader, train_dataset):
    model.train()
    
    if (0 == getloss):
        ############# It is a good idea to regularly keep decreasing the learning rate
        nbrStepsRate = 40
        fraction = []
        lrt = []
        for i in range(1, nbrStepsRate + 1 ):
            r = 0.9**i
            lrt.append(lr*r)
            fraction.append(i/nbrStepsRate)
        v = []
        
        for i in range(nbrStepsRate):
            #print("num_epochs =", num_epochs, "fraction[i] =", fraction[i])
            v = int(num_epochs * fraction[i])
            #print("epoch =", epoch, "v =", v)
            if (epoch == v):   
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrt[i]
                    if (1 == verbose):
                        pass
                        #print("LR=",param_group['lr'])           
             
    ############ Now train
    
    loss_all = 0
    for data in train_loader:
        #print("in for loop of train_loader")
        data = data.to(device)
        #print("train definition, edge_attr =",data.edge_attr.size())
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        label = label.reshape(-1,numTarget)
        label = label[:,targetNbr]
        loss = criterion(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()        
        optimizer.step()
       
    if (1 == verbose):
        print("epoch =", epoch, "  loss =",loss_all / len(train_dataset))
    return loss_all / len(train_dataset)

def evaluate(loader,whichDataset):
    
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            label = data.y.to(device)
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
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
        label = label.reshape(-1,numTarget)
        label = label[:,targetNbr]
        label = label.cpu()
        #print("label =", label)
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
        label = label.reshape(-1,numTarget)
        label = label[:,targetNbr]
        label = label.cpu()
        preds.append(output)
        target.append(label)   
    preds = torch.cat(preds,0)
    target = torch.cat(target,0)
    #print("Datapoints in the validation set =", len(preds))    
    testData = pd.DataFrame({'Target': target, 'Preds': preds})

    return(trainData, testData)

