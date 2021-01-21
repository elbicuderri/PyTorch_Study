# 실행시 인자를 전달하기 위한 library
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_iris

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

net = Net().to(device)

net.load_state_dict(torch.load("torch_iris_model.pt"))

parser = argparse.ArgumentParser(description='Display the logits of the model')
"""
기본 인자로 3을 설정
150개를 다 보고 싶으면
python infer_iris.py -n 150
"""
parser.add_argument('-n', type=int, help='how many display, default=3', default=3)

args = parser.parse_args()

batch = args.n

# 150이 넘으면 에러 발생
assert (batch <= 150), "not over 150"

iris_data = load_iris()

x = iris_data.data

x = x[:batch, :]

x = torch.tensor(x, dtype=torch.float).to(device)

output = net(x)

output = output.cpu().detach().numpy()

print(type(output[0, 0]))/

print(output.shape)

print("\n")

for i in range(batch):
    print(f"{i}th logit: ")
    print(output[i, :])
    print("")
