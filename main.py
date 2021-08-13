import pandas as pd
import torch
from torch.utils.data import random_split
import torch.nn as nn
from torch.utils.data import TensorDataset
#data=torch.tensor(pd.read_csv("water_potability.csv").dropna().values,requires_grad=True)

data=pd.read_csv("heart.csv").dropna()

dataset=TensorDataset(torch.tensor(data.iloc[:,:13].values,dtype=torch.float32),torch.tensor(data.iloc[:,13:].values.reshape([303]),dtype=torch.long))




train , val , test =random_split(dataset,[242,31,30]) 

from torch.utils.data import DataLoader
batch_size = 64
train_dl=DataLoader(train,batch_size,shuffle=True)
val_dl=DataLoader(train,batch_size)
test_dl=DataLoader(test,batch_size)


def acc(outputs,preds): 
    probility,pred=torch.max(preds,dim=1)
    return torch.tensor(torch.sum(outputs==pred).item()/len(pred))

def fit(epoch_size,train_dl,val_dl,model):
    opt=torch.optim.SGD(model.parameters(),lr=1e-5)
    accurcy=[]
    
    for epoch in range(epoch_size):
        #train
        for x,y in train_dl:
            #generate preds
            preds=model(x)
           
            #calculate loss 
            loss=nn.functional.cross_entropy(preds,y)
            
            #graident descent
            loss.backward()
            
            #update weights
            opt.step()
            
            #reset parameters
            opt.zero_grad()
        if(epoch%10000==0):
            print("loss : ", loss.item())
        
        
        for x,y in val_dl:
            preds=model(x)
            accurcy.append(acc(y,preds))
        if(epoch%10000==0):
            print("acc: ",torch.stack(accurcy).mean())
       
            
model=nn.Linear(13,2)

fit(100000,train_dl,val_dl,model)


for x,y in test_dl:
    prob,pred=torch.max(model(x), dim=1)
    print("tahmin: " , pred)
    print("gerçek: ",y)
    print(torch.tensor(torch.sum(y==pred)/len(pred)))


