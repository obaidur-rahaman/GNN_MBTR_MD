import pandas as pd
import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from svglib.svglib import svg2rlg
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_absolute_error, r2_score
import pickle 

def plot_target_vs_avg_edge_attr(dataset):
    
    mean_edge_attr = []
    ti = []
    for i in range(len(dataset)):
        value = dataset[i].edge_attr.mean()
        TI = dataset[i].y
        mean_edge_attr.append(value)
        ti.append(TI)
        
    mean_edge_attr = torch.FloatTensor(mean_edge_attr)
    ti = torch.FloatTensor(ti)
    
    counts, bins = np.histogram(mean_edge_attr, bins=500)
    
    matplotlib.pyplot.scatter(mean_edge_attr, ti)
    plt.show()
    plt.hist(bins[:-1], bins, weights=counts)
    #plt.close()
    
    
def plot_target_vs_features(feature, target):
    plt.rcParams["figure.figsize"] = (10,8)
    matplotlib.pyplot.scatter(feature, target)
    
def plot_results(trainData, testData, target_term, show): 

    target_term = target_term.replace(" ", "_").replace("/", "")
    preds = trainData["Preds"]
    target = trainData["Target"] 
    
    print("Datapoints in the training set =", len(preds))
    #plt.show()
    plt.rcParams["figure.figsize"] = (10,8)
    plt.scatter(target,preds)

    y_train = target.to_numpy()
    y_pred = preds.to_numpy()
    plt.xlim()
    limits = [(min(np.min(y_train), np.min(y_pred))) - 0.2, 0.2 + max(0, (np.max(y_train)), (np.max(y_pred)))]
    plt.xlim(limits)
    plt.ylim(limits)
    infotext = "MAE = {:.3f}\n".format(mean_absolute_error(y_train, y_pred)) + r"$r^2$ = {:.3f}".format(r2_score(y_train, y_pred))
    plt.text(limits[0], limits[1], infotext, bbox={"facecolor": "lightblue", "pad": 5})

    # for test 
    
    preds = testData["Preds"]
    target = testData["Target"] 

    print("Datapoints in the validation set =", len(preds))
    plt.rcParams["figure.figsize"] = (10,8)
    plt.scatter(target,preds)    
    
    y_test = target.to_numpy()
    y_pred = preds.to_numpy()    
    plt.suptitle(target_term, fontsize=30)
    plt.xlabel("%s_DFT"%target_term, fontsize=18)
    plt.ylabel("%s_GNN"%target_term, fontsize=18)        
    infotext2 = "MAE = {:.3f}\n".format(mean_absolute_error(y_test, y_pred)) + r"$r^2$ = {:.3f}".format(r2_score(y_test, y_pred))
    #plt.text(-6, -6, infotext2)
    plt.text(limits[0], 0.8*limits[1], infotext2, bbox={"facecolor": "orange", "pad": 5})

    plt.savefig('../plots/python/%s.png'%target_term)
    if (0 == show):
        plt.close()
    #plt.show()
    #plt.close()
    return(trainData, testData)

def plot_hyper(target_term, nbrRounds):
    
    with open("../results/%s/hyperOpt_param_list_round=%s.pic"%(target_term,nbrRounds), 'rb') as filehandle:
        param_list_all = pickle.load(filehandle)
    with open("../results/%s/hyperOpt_train_loss_list_round=%s.pic"%(target_term,nbrRounds), 'rb') as filehandle:
        train_loss_all = pickle.load(filehandle)
    with open("../results/%s/hyperOpt_val_loss_list_round=%s.pic"%(target_term,nbrRounds), 'rb') as filehandle:
        val_loss_all = pickle.load(filehandle)

    nbrSubPlots = int(round(len(param_list_all)/2 + 0.01))
        
    fig, axs = plt.subplots(nbrSubPlots, 2, figsize=(15,15))   
    for i, ax in enumerate(fig.axes):
        #print(i,ax)
        ax.plot(param_list_all[i], train_loss_all[i], label = "Training loss")
        ax.plot(param_list_all[i], val_loss_all[i], label = "Validation loss")
        ax.legend()
        ax.set_title("param =%s"%i) 
    plt.savefig('../plots/python/%s_hyperParaOpt_round=%s.png'%(target_term,nbrRounds))

def plot_losses(target_term, GNN, MD, MBTR):        
    df_loss = pd.read_csv("../results/%s/loss_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    frac = int(0.25*len(df_loss))
    df_loss = df_loss[frac:]
    ax = df_loss["train_loss"].plot(x="epoch", figsize=(5,5))   
    df_loss.plot(x="epoch", y="validation_loss",kind = "scatter", ax =ax, color='r') 
    ax.legend(['train_loss', 'validation_loss'])
    plt.show()