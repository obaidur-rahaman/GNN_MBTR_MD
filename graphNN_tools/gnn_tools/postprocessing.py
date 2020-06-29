import pandas as pd
import numpy as np
import gnn_tools as gnn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def getLearningCurve(target_term, df_reduced, dataset, split_size, nbrEpochs, param_best):
    df_results = pd.DataFrame(columns = ["training_size", "r2_GNN", "r2_MD", "r2_MDGNN", "MAE_GNN", "MAE_MD", "MAE_MDGNN"])
    index = 0
    x = 1000
    while x<= len(dataset):
        end = x
        print("Training size =", x)       
        dataset_part = dataset[:end]
        df_results.loc[index,"training_size"] = x
        df_part = df_reduced[:end]
        # GET RESULTS FOR gnn AND gnn + md
        for MD in range(2):
            trainData, testData = gnn.fit_GNN(0, 0, target_term, dataset_part, split_size, nbrEpochs, *param_best, MD)            
            #gnn.plot_results(trainData, testData, target_term, show = 1)
            r2 = r2_score(testData["Target"].to_numpy(), testData["Preds"].to_numpy())
            MAE = mean_absolute_error(testData["Target"].to_numpy(), testData["Preds"].to_numpy())  
            if (0 == MD):
                print("GNN only: r2 =", r2, "MAE =", MAE)
                df_results.loc[index,"r2_GNN"] = r2
                df_results.loc[index,"MAE_GNN"] = MAE
                trainData.to_csv("../results/%s/learning_size=%s_train_CNN=1_MD=0.csv" % (target_term, x))
                testData.to_csv("../results/%s/learning_size=%s_test_CNN=1_MD=0.csv" % (target_term, x))
            else:
                print("MDGNN: r2 =", r2, "MAE =", MAE)
                df_results.loc[index,"r2_MDGNN"] = r2
                df_results.loc[index,"MAE_MDGNN"] = MAE
                trainData.to_csv("../results/%s/learning_size=%s_train_CNN=1_MD=1.csv" % (target_term, x))
                testData.to_csv("../results/%s/learning_size=%s_test_CNN=1_MD=1.csv" % (target_term, x))
        # GET RESULTS FOR MD ONLY
        trainData, testData, feat_importances = molecularDescriptorsOnly(df_part, split_size, target_term, 0)
        r2 = r2_score(testData["Target"].to_numpy(), testData["Preds"].to_numpy())
        MAE = mean_absolute_error(testData["Target"].to_numpy(), testData["Preds"].to_numpy())
        print("MD only: r2 =", r2, "MAE =", MAE)
        print("\n")
        df_results.loc[index,"r2_MD"] = r2
        df_results.loc[index,"MAE_MD"] = MAE
        trainData.to_csv("../results/%s/learning_size=%s_train_CNN=0_MD=1.csv" % (target_term, x))
        testData.to_csv("../results/%s/learning_size=%s_test_CNN=0_MD=1.csv" % (target_term, x))
        x *= 2
        index = index + 1
    return(df_results)


def molecularDescriptorsOnly(df, split_size, target_term, show):
    df = df.fillna(0)
    y = df.loc[:,target_term]
    X = df.loc[:,'NbrAtoms':]
    X1 = X
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1-split_size), random_state=42)
    
    #model = LinearRegression().fit(X_train, y_train)
    model = RandomForestRegressor(n_estimators=300, max_depth=7).fit(X_train, y_train)
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    trainData = pd.DataFrame({'Target': y_train, 'Preds': preds_train})
    testData = pd.DataFrame({'Target': y_test, 'Preds': preds_test})        
    if (1 == show):
        gnn.plot_results(trainData, testData, target_term, show = 1)
        plt.show()
        plt.close()
    
    feat_importances = pd.Series(model.feature_importances_, index=X1.columns) 
    if (1 == show):
        feat_importances.nlargest(10).sort_values().plot(kind='barh')
        plt.show()
    feat_importances = feat_importances.sort_values(ascending=False)
    return(trainData, testData, feat_importances)
