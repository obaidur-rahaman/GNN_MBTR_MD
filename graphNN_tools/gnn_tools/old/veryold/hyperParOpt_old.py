
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
import sys
sys.path
sys.path.append('/home/ge35gen/research/organic_semiconductor/graphNN_tools/')
import gnn_tools as gnn

#parameter_ranges = [[8, 128],[4,128],[4,128],[4,128],[4,128],[4,128]]
#nbrTrials = 50
#nbrEpochs = 7

def fit_hyperParameters(dataset1, parameter_ranges, nbrTrials, nbrEpochs):

    global dataset
    dataset = dataset1
    splitRatio = 0.5
    results = create_rand(parameter_ranges, nbrTrials, 1, nbrEpochs) #parameter_ranges,nbrTrials, run, nbrEpochs
    results = results.sort_values(results.columns[-1])
    results = results.reset_index(drop= True)
    #print(results)
    cut = int(splitRatio*len(results))
    best_results = results[:cut]
    #print(best_results)
    param_best = best_results[:1].values.flatten().tolist()
    param_best_5 = best_results[:5].values.tolist()
    param_best.pop()
    '''
    # Now we will get a machine learning model to predict the best parameters
    parameter_ranges_shorter = []
    for i in range(len(parameter_ranges)):
        list = []
        list.append(min(best_results[i]))
        list.append(max(best_results[i]))
        parameter_ranges_shorter.append(list)
    # First fit the model
    y = best_results.iloc[:,-1]
    X = best_results.iloc[:,:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Ridge(alpha = 2.0, normalize=True)
    model.fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    matplotlib.pyplot.scatter(y_train,preds_train)
    matplotlib.pyplot.scatter(y_test,preds_test)
    # Now generate a diverse dataset with a lot of possible combinations
    # Then predict the best set of parameters based on the model
    large_results = create_rand(parameter_ranges_shorter, 5000, 0, 0)
    preds = model.predict(large_results)
    large_results["loss"] = preds
    large_results = large_results.sort_values(large_results.columns[-1])
    large_results = large_results.reset_index(drop= True)
    param_best_ml = large_results[:1].values.flatten().tolist()
    param_best_ml.pop()
    '''
    return(param_best, param_best_5) #, param_best_ml)


# This creates a dataframe with random values of the parameters and optionally
# calculates the loss
def create_rand(parameter_ranges,nbrTrials, run, nbrEpochs):
    results = pd.DataFrame()
    for trial in range(nbrTrials):
        #print("trial =", trial)
        list = []
        for p in range(len(parameter_ranges)):
            #print("P = ", p)
            p1 = parameter_ranges[p][0]
            p2 = parameter_ranges[p][1]
            #print(p1,p2)
            randValue = random.randrange(p1, p2)
            #print(randValue)
            results.loc[trial, p] = int(randValue) 
        if (1 == run):
            print("trial =", trial)
            list = results.loc[trial:trial, 0:(len(parameter_ranges)-1)].values.tolist()
            #print("list =", list)
            array1 = np.asarray(list).flatten().astype(int)        
            array2 = np.asarray([dataset,0.75, nbrEpochs])
            full_array = np.concatenate((array2, array1)).flatten()
            #print(full_array)
            val_acc = gnn.fit_GNN(0,*full_array) 
            results.at[trial, len(parameter_ranges)] = val_acc
    return(results)




