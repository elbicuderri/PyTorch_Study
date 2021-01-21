import numpy as np

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

net = Net().to(device)

net.load_state_dict(torch.load("torch_iris_model.pt"))

# named_parameters(): weight의 이름 정보도 얻을 수 있는 함수
for name, weight in net.named_parameters():
    print(name, weight.shape)

with open("torch_iris_weight.bin", "wb") as f:
    # 처음 40bytes는 float32(0) * 10개를 채워준다.
    (np.asarray([0 for _ in range(10)], dtype=np.float32)).tofile(f)

    for weight in net.parameters():
        """
        w = np.asarray(weight.cpu().detach(), dtype=np.float32)
        print(weight.shape, isinstance(w, np.ndarray))
        print(type(w[0]))
        """
        (np.asarray(weight.cpu().detach(), dtype=np.float32)).tofile(f)
