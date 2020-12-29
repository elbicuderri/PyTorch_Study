import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from statistics import mean
import numpy as np
# from torchviz import make_dot
# from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# def ResidualBlock(in_channels, out_channels, kernel_size, padding, stride, bias=False):
#     return nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            # kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            # kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
#             nn.BatchNorm2d(out_channels),        
#         )

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
        )

        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
        )

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)

        self.block21 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False)

        self.block31 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.block32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.avg_pool = nn.AvgPool2d(8)
        # self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        batch = x.size(0)
        out0 = self.conv0(x)

        res11 = out0
        out11 = self.block11(out0)
        out11 += res11
        out11 = self.relu(out11)

        res12 = out11
        out12 = self.block12(out11)
        out12 += res12
        out12 = self.relu(out12)

        res21 = self.conv2(out12)
        out21 = self.block21(out12)
        out21 += res21
        out21 = self.relu(out21)

        res22 = out21
        out22 = self.block22(out21)
        out22 += res22
        out22 = self.relu(out22)

        res31 = self.conv3(out22)
        out31 = self.block31(out22)
        out31 += res31
        out31 = self.relu(out31)

        res32 = out31
        out32 = self.block32(out31)
        out32 += res32
        out32 = self.relu(out32)

        out4 = self.avg_pool(out32)
        # out3 = self.flatten(out3)
        out4 = out4.view(batch, -1)
        out = self.fc(out4)

        return out

model = SimpleResNet().to(device)

summary(model, input_size=(3, 32, 32))

# InTensor = Variable(torch.randn(1, 3, 32, 32)).to(device)
# make_dot(model(InTensor), params=dict(model.named_parameters())).render("model", format="png")

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

loss_dict = {}
val_loss_dict = {}
train_step = len(train_loader)
val_step = len(valid_loader)
epochs = 5

for i in range(1, epochs + 1):
    loss_list = [] # losses of i'th epoch
    for train_step_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        output = model(img)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if ((train_step_idx+1) % 100 == 0):
            print(f"Epoch [{i}/{epochs}] Step [{train_step_idx + 1}/{train_step}] Loss: {loss.item():.4f}")

    loss_dict[i] = loss_list

    val_loss_list = []
    for val_step_idx, (val_img, val_label) in enumerate(valid_loader):
        with torch.no_grad():
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            
            model.eval()
            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)

        val_loss_list.append(val_loss.item())

    val_loss_dict[i] = val_loss_list
    
    torch.save({
        f"epoch": i,
        f"model_state_dict": model.state_dict(),
        f"optimizer_state_dict": optimizer.state_dict(),
        f"loss": loss},
            f"cifar10_epoch_{i}.ckpt")

    print(f"Epoch [{i}] Train Loss: {mean(loss_dict[i]):.4f} Val Loss: {mean(val_loss_dict[i]):.4f}")
    print("========================================================================================")

torch.save(model.state_dict(), 'resnet.pt')