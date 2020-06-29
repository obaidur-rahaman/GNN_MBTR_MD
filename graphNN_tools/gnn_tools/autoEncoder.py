import numpy as np
import pandas as pd
import sys
sys.path
sys.path.append('/home/ge35gen/research/organic_semiconductor/graphNN_tools/')
import gnn_tools as gnn
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

def autoEncode():
    print("#############  Autoencoding MBTR")
    # Make a dataset
    global criterion, model, optimizer, nbrFeatures, device, mbtr_latent, nbrEpochs
    df = pd.read_pickle("../data/mbtr.pic")
    nbrFeatures = len(df.columns)
    l = int(round(0.8*(len(df))))
    df = df.sparse.to_coo()
    dataset = MBTI_dataset(df)
    
    del df
    
    #all_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    trainLoader = DataLoader(dataset[:l], batch_size=32)
    valLoader = DataLoader(dataset[l:], batch_size=32)

    #  use gpu if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AutoEncoder(input_shape=nbrFeatures).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)

    # mean-squared error loss
    criterion = nn.MSELoss()
    nbrEpochs = 100
       
    for epoch in range(nbrEpochs):
        train_loss = train(epoch, trainLoader)
        for ep in range(9, nbrEpochs, 10):             
            if (epoch == ep):
                eval_loss = evaluate(epoch, valLoader)
                print("Epoch :", epoch + 1, "/", nbrEpochs, "Training loss =", round(1000000 * train_loss,6), "Evaluation loss=", round(1000000 * eval_loss,6))
    
    mbtr_latent = get_latent_code(trainLoader, valLoader)
    mbtr_latent = mbtr_latent.drop(['mbtr'], axis=1)
    mbtr_latent['mbtr']= mbtr_latent.values.tolist()
    mbtr_latent = mbtr_latent[['mbtr']]
    mbtr_latent.to_pickle("../data/mbtr_latent.pic")

class MBTI_dataset(Dataset):
    def __init__(self, df):
        #df = df.fillna(0)
        #data = torch.tensor(df.values, dtype=torch.sparse_coo)
        values = df.data
        indices = np.vstack((df.row, df.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = df.shape

        data = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.data = data.to_dense()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(epoch, trainLoader):
    loss = 0
    for batch_features in trainLoader:
        torch.cuda.empty_cache()
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, nbrFeatures).to(device)      
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()   
        # compute reconstructions
        code, outputs = model(batch_features)  
        del code    
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features) 
        del outputs     
        # compute accumulated gradients
        train_loss.backward()      
        # perform parameter update based on current gradients
        optimizer.step()     
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()   
    # compute the epoch training loss
    loss = loss / len(trainLoader)    
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, nbrEpochs, 1000000 * loss))
    torch.cuda.empty_cache()
    return(loss)

def evaluate(epoch, valLoader):
    loss = 0
    with torch.no_grad():
        for batch_features in valLoader:
            torch.cuda.empty_cache()
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, nbrFeatures).to(device)      
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()   
            # compute reconstructions
            code, outputs = model(batch_features)
            code = code.data.detach().cpu()
            code = code.data.detach().cpu()
            del code      
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)      
            # perform parameter update based on current gradients
            optimizer.step()     
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()   
    # compute the epoch training loss
    loss = loss / len(valLoader)    
    # display the epoch training loss
    #print("Validation loss epoch : {}/{}, loss = {:.6f}".format(epoch + 1, nbrEpochs, loss))
    return(loss)

def get_latent_code(trainLoader, valLoader):
    mbtr_latent = pd.DataFrame({'mbtr' : []}) 
    with torch.no_grad():
        for loader in [trainLoader, valLoader]:
            for batch_features in loader:
                torch.cuda.empty_cache()
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features = batch_features.view(-1, nbrFeatures).to(device)      
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()   
                # compute reconstructions
                code, outputs = model(batch_features)  
                code1 = code.data.detach().cpu().numpy()  
                #print(code1)  
                latent = pd.DataFrame(code1)
                #print("latent = ", len(latent), len(latent.columns))
                mbtr_latent = pd.concat([mbtr_latent, latent], ignore_index=True,sort=False)
                #print('mbtr_latent', len(mbtr_latent), len(mbtr_latent.columns))
    return(mbtr_latent)

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=1024
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1024, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        #features = features.to_dense()
        #features[features != features] = 0
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return (code, reconstructed)