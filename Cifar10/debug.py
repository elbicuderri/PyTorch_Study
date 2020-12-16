import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from torchviz import make_dot
from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 32
batch_size = 64

transform = transforms.Compose([
 transforms.Pad(4),
 transforms.RandomHorizontalFlip(),
 transforms.RandomCrop(32),
 transforms.ToTensor(),
 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # only can do to tensor so keep order
])

train_dataset = datasets.CIFAR10('C:\data/cifar10', train=True, download=True, transform=transform)

train_loader = DataLoader(
    dataset= train_dataset,
    batch_size=batch_size,
    shuffle=True)

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

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
        #out2 = nn.ReLU(out2, inplace=True)

        res3 = self.conv3(out2)
        out3 = self.block31(out2)
        out3 = self.block32(out3)
        out3 += res3
        out3 = self.relu(out3)
        #out3 = nn.ReLU(out3, inplace=True)

        out3 = self.avg_pool(out3)
        out3 = self.flatten(out3)
        #out3 = nn.Flatten()(out3)
        out = self.fc(out3)

        return out

model = MnistModel()

InTensor = Variable(torch.randn(1, 3, 32, 32))
make_dot(model(InTensor), params=dict(model.named_parameters())).render("model", format="png")

summary(model, input_size=(3, 32, 32))
