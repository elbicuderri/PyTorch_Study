import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from torchviz import make_dot
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from statistics import mean
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

transform = transforms.Compose([
 transforms.Pad(4),
 transforms.RandomHorizontalFlip(),
 transforms.RandomCrop(32),
 transforms.ToTensor(),
 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # only can do to tensor so keep order
])

train_dataset = datasets.CIFAR10('C:\data/cifar10/train/', train=True, download=True, transform=transform)

train_loader = DataLoader(
    dataset= train_dataset,
    batch_size=batch_size,
    shuffle=True)

valid_dataset = datasets.CIFAR10(root='C:\data/cifar10/test/',
                                            train=False, 
                                            transform=transforms.ToTensor())

valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.block11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)

        self.block21 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32)
        )

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False)

        self.block31 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.block32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.avg_pool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.block11(out0)
        out1 = self.block12(out1)

        res2 = self.conv2(out1)
        out2 = self.block21(out1)
        out2 = self.block22(out2)
        out2 += res2
        out2 = self.relu(out2)

        res3 = self.conv3(out2)
        out3 = self.block31(out2)
        out3 = self.block32(out3)
        out3 += res3
        out3 = self.relu(out3)

        out3 = self.avg_pool(out3)
        out3 = self.flatten(out3)
        out = self.fc(out3)

        return out

model = SimpleResNet().to(device)

InTensor = Variable(torch.randn(1, 3, 32, 32)).to(device)
make_dot(model(InTensor), params=dict(model.named_parameters())).render("model", format="png")

summary(model, input_size=(3, 32, 32))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(
    optimizer=optimizer, 
    mode='min',
    factor=0.5,
    patience=2
    )

loss_dict = {}
val_loss_dict = {}
train_step = len(train_loader)
val_step = len(valid_loader)
epochs = 50

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

    with torch.no_grad():
        val_loss_list = []
        for val_step_idx, (val_img, val_label) in enumerate(valid_loader):
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            
            model.eval()
            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)
            scheduler.step(val_loss)

            val_loss_list.append(val_loss.item())

        val_loss_dict[i] = val_loss_list
        
        # best_loss = mean(val_loss_dict[i])
        torch.save({
            f"epoch": i,
            f"model_state_dict": model.state_dict(),
            f"optimizer_state_dict": optimizer.state_dict(),
            f"loss": loss},
             f"checkpoint/resnet_cifar10_checkpoint_epoch_{i}.ckpt")


    print(f"Epoch [{i}] Train Loss: {mean(loss_dict[i]):.4f} Val Loss: {mean(val_loss_dict[i]):.4f}")
    print("========================================================================================")


torch.save(model.state_dict(), 'resnet.pt')