import numpy as np

# skleran에서 제공해주는 iris datasets, train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
# load IRIS dataset
iris_data = load_iris()
x = iris_data.data
y = iris_data.target

train_X, test_X, train_y, test_y = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True, 
                                                    random_state=31)

train_X = torch.tensor(train_X, dtype=torch.float).to(device) ## torch.float --> 32bit
test_X = torch.tensor(test_X, dtype=torch.float).to(device)
train_y = torch.tensor(train_y, dtype=torch.long).to(device) ## torch.long --> 64bit
test_y = torch.tensor(test_y, dtype=torch.long).to(device)

net = Net().to(device)

criterion = nn.CrossEntropyLoss()# cross entropy loss

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(1000):
    out = net(train_X)
    loss = criterion(out, train_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print (f"number of epoch: {epoch} loss: {loss.item()}")

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)

torch.save(net.state_dict(), "torch_iris_model.pt")