import pickle
import pandas as pd
import random
import numpy as np
import gnn_tools as gnn
import subprocess
import sys

def create_rand(target_term, split_size, parameter_ranges,nbrTrials, nbrEpochs, GNN, MD, MBTR):
    with open("./hyper/dataset.pic", 'rb') as f:
        dataset = pickle.load(f)
    for trial in range(nbrTrials):
        results = pd.DataFrame()
        results = results.astype('object')
        #print("trial =", trial)
        for p in range(len(parameter_ranges)):
            #print("P = ", p)
            p1 = parameter_ranges[p][0]
            p2 = parameter_ranges[p][1]
            #print(p1,p2)
            # Ignore the learning rate and randomize the others
            if 0 == p:
                results.loc[trial, p] = random.uniform(p1, p2)
            else:
                if p1 == p2:
                    randValue = p1
                else:
                    randValue = random.randrange(p1, p2+1)
                #print(randValue)
                results.loc[trial, p] = int(randValue) 
                results[p] = results[p].astype(int)
                #print(results)
        
        #print("trial =", trial)
        my_list = results.loc[trial:trial, 0:(len(parameter_ranges)-1)].values.tolist()
        my_list = my_list[0]
        lr = my_list[0]
        my_list = [ int(x) for x in my_list ]
        my_list[0] = lr 
        train_acc, val_acc = gnn.fit_GNN(1, 0, target_term, dataset,split_size, nbrEpochs, *my_list, GNN, MD, MBTR)
        results.at[trial, len(parameter_ranges)] = round(val_acc,5)
        print(results.loc[trial:trial, :].to_string(header=False))
        # Now store the results
        try:
            results_old = pd.read_csv("../results/%s/hyper_random_results.csv"%target_term, header = None)
            #print("results_old")
            #print(results_old)
            all_results = pd.concat([results_old, results], ignore_index=True, axis=0)
            all_results.to_csv("../results/%s/hyper_random_results.csv"%target_term, index=False, header = False)
            #subprocess.run(["mv", "./hyper_random/results_temp.csv", "./hyper_random/results.csv"]) 
        except:
            results.to_csv("../results/%s/hyper_random_results.csv"%target_term, index=False, header =False)
            #print("results")
            #print(results)
        #print("line 2")

i = sys.argv[1]

with open("../results/%s/Hyper_arguments_list.pic"%i, 'rb') as f:
    argument_list = pickle.load(f)
create_rand(*argument_list)
