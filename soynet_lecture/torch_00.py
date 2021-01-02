import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=16,
                              kernel_size=3, padding=1, stride=1, bias=False)      
        self.batchnorm = nn.BatchNorm2d(16, affine=True)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        batch = x.size(0) # x.shape[0]
        out0 = self.conv(x)
        out1 = self.batchnorm(out0)
        out2 = self.relu(out1)
        out3 = self.avg_pool(out2)
        out4 = out3.view(batch, -1)
        out5 = self.fc(out4)

        return out5

model = SimpleModel().to(device)

batch_size = 32
epochs = 3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
 transforms.ToTensor(), # 0 ~ 1
 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
]) # output[channel] = (input[channel] - mean[channel]) / std[channel]

train_dataset = datasets.CIFAR10('train/',
                                 train=True,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

valid_dataset = datasets.CIFAR10(root='test/',
                                            train=False, 
                                            download=True,
                                            transform=transform)

valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

for epoch in range(1, epochs + 1):
    for img, label in train_loader:
        model.train()
        img = img.to(device)
        label = label.to(device)
        
        output = model(img)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[{epoch}/{epochs}] finished")
    print('==================')

torch.save(model.state_dict(), 'cifar10_model.pt')



# from torchsummary import summary
# from torchviz import make_dot
# from torch.autograd import Variable