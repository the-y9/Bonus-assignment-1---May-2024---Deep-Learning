import numpy as np
#%% 1
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()
df.data.shape
# 8

#%% 2

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(df.data,df.target,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xtrain = ss.fit_transform(xtrain)
xtest = ss.transform(xtest)

last_test_point = xtest[-1]
min_v = np.min(last_test_point)
max_v = np.max(last_test_point)
f"{min_v:2f},{max_v:2f}"

# Out[45]: '-0.921138,0.604455'
#%% 3
import torch
torch.manual_seed(42)

xtrain_tensor = torch.tensor(xtrain,dtype=torch.float32)
xtest_tensor = torch.tensor(xtest,dtype=torch.float32)
ytrain_tensor = torch.tensor(ytrain,dtype=torch.float32).reshape(-1,1)
ytest_tensor = torch.tensor(ytest,dtype=torch.float32).reshape(-1,1)

xtrain_tensor.shape,ytrain_tensor.shape
# Out[46]: (torch.Size([16512, 8]), torch.Size([16512, 1]))
#%% 4

import torch.nn as nn
torch.manual_seed(42)

class RegressionANN(nn.Module):
    def __init__(self,hln):
        super(RegressionANN,self).__init__()
        self.input_layer =nn.Linear(8, hln)
        self.hidden_layer = nn.Linear(hln, 1)
        
    def forward(self,x):
        x = self.input_layer(x)
        x = torch.relu(self.hidden_layer(x))
        return x
#%%    
model = RegressionANN(hln=16)
initial_bias = model.hidden_layer.bias.data
print(initial_bias)
# tensor([0.2272])

#%% 5

import torch.optim as optim

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
outputs = model(xtrain_tensor)
initial_loss = loss_function(outputs,ytrain_tensor)
print(initial_loss)
# tensor(4.4983, grad_fn=<MseLossBackward0>)

#%% 6

eno,l=[],[]
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(xtrain_tensor)
    loss = loss_function(predictions, ytrain_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 ==0:
        eno.append(epoch+1)
        l.append(float(loss))
        
import matplotlib.pyplot as plt
plt.figure()
plt.plot(eno,l)
plt.xlabel('Epoch Numbers')    
plt.ylabel('Loss')
plt.show()
print(l[-1])
# 0.46388015151023865
#%% 7
torch.manual_seed(42)
model = RegressionANN(hln = 64)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
outputs = model(xtrain_tensor)

eno,l=[],[]
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(xtrain_tensor)
    loss = loss_function(predictions, ytrain_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 ==0:
        eno.append(epoch+1)
        l.append(float(loss))
        
import matplotlib.pyplot as plt
plt.figure()
plt.plot(eno,l)
plt.xlabel('Epoch Numbers')    
plt.ylabel('Loss')
plt.show()
print(l[-1])
# 0.46233490109443665
#%%















