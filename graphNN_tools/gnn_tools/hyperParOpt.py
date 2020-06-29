import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, RidgeCV, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
import sys
sys.path
sys.path.append('/home/ge35gen/research/organic_semiconductor/graphNN_tools/')
import gnn_tools as gnn
from statistics import median
import pickle
import subprocess
import time
import threading
from subprocess import Popen
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
#parameter_ranges = [[0.005, 0.005],[16, 16], [24, 24], [36, 36], [64, 200], [1,1], [0,4], [10, 250]]
#nbrTrials = 50
#nbrEpochs = 7

def fit_hyperParameters_simple(target_term, split_size, nbrEpochs, nbrGrid, param_range, param_initial, GNN, MD, MBTR, show):
    print("######################### ", target_term, " HYPER PARAMETER OPTIMIZATION")
    param_best = param_initial # initialize parameter set  
    target_term = target_term.replace(" ", "_").replace("/", "")
    for  i in range(2):
        nbrRounds = i + 1
        print("############## ROUND =", nbrRounds)    
        param_best = OptimizeParam(nbrRounds, target_term, split_size, nbrEpochs, nbrGrid, param_range, param_best, GNN, MD, MBTR, show)      
    return(param_best)

def OptimizeParam(nbrRounds, target_term, split_size, nbrEpochs, nbrGrid, param_range, param_best, GNN, MD, MBTR, show):   
    train_loss_all = []
    val_loss_all = []
    param_list_all = []
    # Start all the jobs in parallel
    
    for i in range(len(param_range)):
        train_loss = []
        val_loss = []
        param_list = []
        low = param_range[i][0]
        high = param_range[i][1]
        if (low == high):
            print("Skipping param", i, " because lowest = highest")
            train_loss_all.append(0)
            val_loss_all.append(0)
            param_list_all.append(low)
        else:
            print("Param", i, ": lowest =", low, "highest =", high)
            
            process = []
            for j in range(nbrGrid):
                p = low + (j*(high-low))/(nbrGrid-1) 
                if (i > 0):
                    p = int(p)
                #print("param", i, "=", p)
                param_best[i] = p
                param_list.append(p) 

                # First write down the arguments
                argument_list = [i, j, p, target_term, split_size, nbrEpochs, nbrGrid, param_best, GNN, MD, MBTR, show]
                with open("./hyper/Hyper_arguments_list_%s.pic"%j, 'wb') as filehandle:
                    pickle.dump(argument_list, filehandle)
                # Now run the job
                proc = Popen("python3 /home/ge35gen/research/organic_semiconductor/graphNN_tools/gnn_tools/hyper_parallel_run_simple.py %s"%j, shell=True, stdin=None, stderr=None, close_fds=True)
                #subprocess.run(["python3", "hyper_parallel_run.py", "&"])
                #print("proc =", proc)
                process.append(proc)
            
            # Now keep checking if all the jobs are finished
            #print("process =", process)
            check_job_finished(process, nbrGrid)
            # Now read the results
            for j in range(nbrGrid):
                with open("./hyper/train_loss_%s.pic"%j, 'rb') as f:
                        trainLoss = pickle.load(f)
                with open("./hyper/val_loss_%s.pic"%j, 'rb') as f:
                        valLoss = pickle.load(f)

                train_loss.append(trainLoss)
                val_loss.append(valLoss)

            print("Hyper parameter optimization for param =", i, "is done!")
            
            with open("./hyper/Hyper_parallel_train_loss_%s.pic"%i, 'wb') as filehandle:
                pickle.dump(train_loss, filehandle)        
            with open("./hyper/Hyper_parallel_val_loss_%s.pic"%i, 'wb') as filehandle:
                pickle.dump(val_loss, filehandle) 
            with open("./hyper/Hyper_parallel_param_list_%s.pic"%i, 'wb') as filehandle:
                pickle.dump(param_list, filehandle) 

            # Now find the parameter that produced the lowest validation loss
            #print(loss_list.index(min(loss_list)))
            index = val_loss.index(min(val_loss))
            param_best[i] = param_list[index]
            
            if (1 == show) and (low != high):
                plt.plot(param_list, train_loss, label = "Training loss") 
                plt.plot(param_list, val_loss, label = "Validation loss")
                plt.legend()
                plt.show()


            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)
            param_list_all.append(param_list)
     
    # Now save the results
    with open("../results/%s/hyperOpt_param_list_round=%s.pic"%(target_term,nbrRounds), 'wb') as filehandle:
        pickle.dump(param_list_all, filehandle)
    with open("../results/%s/hyperOpt_train_loss_list_round=%s.pic"%(target_term,nbrRounds), 'wb') as filehandle:
        pickle.dump(train_loss_all, filehandle)
    with open("../results/%s/hyperOpt_val_loss_list_round=%s.pic"%(target_term,nbrRounds), 'wb') as filehandle:
        pickle.dump(val_loss_all, filehandle)
    
    np.savetxt("../results/%s/best_params_round=%s.csv"%(target_term,nbrRounds), param_best, delimiter=",", fmt='%s')
    print("Best set of parameters =", param_best)
    return(param_best)

def fit_hyperParameters_random(hyper_batch_size, target_term1, split_size1, parameter_ranges, nbrTrials, nbrEpochs, GNN1, MD1, MBTR1):
    global target_term, split_size, GNN, MD, MBTR
    target_term = target_term1
    split_size = split_size1
    GNN = GNN1
    MD = MD1
    MBTR = MBTR1
    #Clean up previous results
    subprocess.run(["rm", "../results/%s/hyper_random_results.csv"%target_term])
    # We can run each trial in parallel in batches
    print("Hyperparameter optimization by random sampling in ", hyper_batch_size, "parallel batches")
    new_nbrTrial = int(nbrTrials/hyper_batch_size)
    # Write down the arguments
    argument_list = [target_term, split_size, parameter_ranges, new_nbrTrial, nbrEpochs, GNN, MD, MBTR]
    with open("../results/%s/Hyper_arguments_list.pic"%target_term, 'wb') as filehandle:
        pickle.dump(argument_list, filehandle)
    p = []    
    for batch in range(hyper_batch_size):
        print("Batch =", batch)
        proc = Popen("python3 /home/ge35gen/research/organic_semiconductor/graphNN_tools/gnn_tools/hyper_parallel_run_random.py %s"%target_term, shell=True, stdin=None, stderr=None, close_fds=True)
        p.append(proc)

    # Now keep checking if all the jobs are finished
    check_job_finished(p, hyper_batch_size)
    # Now read in the results
    results = pd.read_csv("../results/%s/hyper_random_results.csv"%target_term, header = None)
    results = results.sort_values(results.columns[-1])
    results = results.reset_index(drop=True)
    param_best = results.loc[0:0,:].values.tolist()[0]
    lr = param_best[0]
    param_best = [ int(x) for x in param_best ]
    param_best[0] = lr
    param_best.pop()
    param_best_5 = results[:5]
    print("Best set of parameters found by random search:", param_best)
    return(param_best, param_best_5)

def check_job_finished(p, n):   
    while True:
        exit_codes = []
        #print("p =", p, "n =", n)
        for batch in range(n):
            exit_codes.append(p[batch].wait()) 
            
        print("exit_codes =", exit_codes)
        if all(c == 0 for c in exit_codes):
            break
        else:
            print("Waiting for the jobs to be finished")
            time.sleep(10)

def fit_ML_hyper_rand(target_term, parameter_ranges):
    splitRatio = 0.5
    results = pd.read_csv("../results/%s/hyper_random_results.csv"%target_term, header = None)
    results = results.sort_values(results.columns[-1])
    results = results.reset_index(drop= True)

    #print(best_results)
    param_best = results[:1].values.flatten().tolist()
    lr = param_best[0]
    param_best = [ int(x) for x in param_best ]
    param_best[0] = lr
    param_best_5 = results[:5] #.values.tolist()
    param_best.pop()
    
    # Now save the results
    np.savetxt("../results/%s/best_params_random.csv"%target_term, param_best, delimiter=",", fmt='%s')
    np.savetxt("../results/%s/best_params_random_5.csv"%target_term, param_best_5, delimiter=",", fmt='%s')
    print("Best set of parameters =", param_best)
    print("5 best sets of parameters =", param_best_5)

    # We are not interested in the cases with high validation accuracy
    cut = int(splitRatio*len(results))
    best_results = results[:cut]

    # We can plot the loss against each hyper parameter to see its significance
    nbrSubPlots = int(round(len(parameter_ranges)/2 + 0.01))   
    fig, axs = plt.subplots(nbrSubPlots, 2, figsize=(15,15))
    for i in range(8):
        #print(i,ax)
        best_results.boxplot(column = [8], by = [i], grid =False, ax = fig.axes[i])
    plt.show()

    # Now we will get a machine learning model to predict the best parameters
    parameter_ranges_shorter = []
    for i in range(len(parameter_ranges)):
        short_list = []
        short_list.append(min(best_results[i]))
        short_list.append(max(best_results[i]))
        parameter_ranges_shorter.append(short_list)
    
    # First fit the model
    y = best_results.iloc[:,-1]
    X = best_results.iloc[:,:-1]
    X1 = X
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #print("change 3")
    model = RandomForestRegressor(n_estimators=1000, max_depth=5).fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    plt.scatter(y_train,preds_train)
    plt.scatter(y_test,preds_test)
    
    lmin = min(np.min(y_train), np.min(preds_train))
    lmax = max(np.max(y_train), np.max(preds_train))
    #print(lmin, lmax)
    limits = [ lmin - (lmax-lmin)*0.2, lmax + (lmax-lmin)*0.2]
    plt.xlim(limits)
    plt.ylim(limits)
    infotext = "MAE = {:.3f}\n".format(mean_absolute_error(y_train, preds_train)) + r"$r^2$ = {:.3f}".format(r2_score(y_train, preds_train))
    plt.text(limits[0], limits[1], infotext, bbox={"facecolor": "lightblue", "pad": 5})
    infotext2 = "MAE = {:.3f}\n".format(mean_absolute_error(y_test, preds_test)) + r"$r^2$ = {:.3f}".format(r2_score(y_test, preds_test))
    plt.text(limits[0], 0.8*limits[1] + 0.2*limits[0], infotext2, bbox={"facecolor": "orange", "pad": 5})
    plt.show()
    
    feat_importances = pd.Series(model.feature_importances_, index=X1.columns) 
    feat_importances.nlargest(10).sort_values().plot(kind='barh')
    plt.show()
    
    # Now generate a diverse dataset with a lot of possible combinations
    # Then predict the best set of parameters based on the model
    large_results = create_rand_ML(parameter_ranges_shorter, 2000)
    preds = model.predict(large_results)
    large_results["loss"] = preds
    large_results = large_results.sort_values(large_results.columns[-1])
    large_results = large_results.reset_index(drop= True)
    param_best_ml = large_results[:1].values.flatten().tolist()
    param_best_ml.pop()
    lr = param_best_ml[0]
    param_best_ml = [ int(x) for x in param_best_ml ]
    param_best_ml[0] = lr
    
    return(param_best_ml) #, param_best_ml)


# This creates a dataframe with random values of the parameters and optionally
# calculates the loss
def create_rand_ML(parameter_ranges,nbrTrials):
    results = pd.DataFrame()
    results = results.astype('object')
    for trial in range(nbrTrials):
        #print("trial =", trial)
        for p in range(len(parameter_ranges)):
            #print("P = ", p)
            p1 = parameter_ranges[p][0]
            p2 = parameter_ranges[p][1]
            #print(p1,p2)
            # Ignore the learning rate and randomize the others
            if 0 == p:
                results.loc[trial, p] = (p1 + p2) / 2
            else:
                if p1 == p2:
                    randValue = p1
                else:
                    randValue = random.randrange(p1, p2+1)
                #print(randValue)
                results.loc[trial, p] = int(randValue) 
                results[p] = results[p].astype(int)
                #print(results)
    return(results)





