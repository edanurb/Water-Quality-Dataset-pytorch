import pandas as pd
import torch
from torch.utils.data import random_split
import torch.nn as nn
from torch.utils.data import TensorDataset
#data=torch.tensor(pd.read_csv("water_potability.csv").dropna().values,requires_grad=True)

data=pd.read_csv("water_potability.csv").dropna()

dataset=TensorDataset(torch.tensor(data.iloc[:,:9].values,dtype=torch.float32),torch.tensor(data.iloc[:,9:].values.reshape([2011]),dtype=torch.long))


a=data.iloc[:,:9].values


train , val , test =random_split(dataset,[1407,402,202]) 

from torch.utils.data import DataLoader
batch_size = 64
train_dl=DataLoader(train,batch_size,shuffle=True)
val_dl=DataLoader(train,batch_size)


def acc(outputs,preds): 
    probility,pred=torch.max(preds,dim=1)

    return torch.tensor(torch.sum(outputs==pred)).item()/len(outputs)

def fit(epoch_size,train_dl,val_dl,model,loss_func):
    opt=torch.optim.SGD(model.parameters(),lr=1e-6)
    accurcy=0
    for epoch in range(epoch_size):
        #train
        for x,y in train_dl:
            #generate preds
            preds=model(x)
           
            #calculate loss 
            #print(preds)
            #print(y)
            loss=loss_func(preds,y)
            
            #graident descent
            loss.backward()
            
            #update weights
            opt.step()
            
            #reset parameters
            opt.zero_grad()
        
        for x,y in val_dl:
            preds=model(x)

            accurcy=acc(y,preds)
        print(accurcy)     
            
model=nn.Linear(9,2)
loss_func=nn.functional.cross_entropy

fit(5,train_dl,val_dl,model,loss_func)


