import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from statistics import mean

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class ConvTransposeModel(nn.Module):
    def __init__(self):
        super(ConvTransposeModel, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm2d(3)
        
        self.batchnorm2 = nn.BatchNorm2d(4)
        
        self.batchnorm3 = nn.BatchNorm2d(3)
        
        self.batchnorm4 = nn.BatchNorm2d(10)
        
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=3,
                               kernel_size=3,
                               padding=1,
                               stride=2,
                               bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=3,
                               out_channels=4,
                               kernel_size=3,
                               padding=1,
                               stride=2,
                               bias=False)
        
        self.convtranspose1 = nn.ConvTranspose2d(in_channels=4,
                                                 out_channels=3,
                                                 kernel_size=3,
                                                 padding=1,
                                                 stride=2,
                                                 output_padding=1,
                                                 bias=False)
        
        self.convtranspose2 = nn.ConvTranspose2d(in_channels=3,
                                                 out_channels=10,
                                                 kernel_size=3,
                                                 padding=1,
                                                 stride=2,
                                                 output_padding=1,
                                                 bias=False)
        
        self.global_avg_pool =nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(10, 10)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_batchnorm = self.batchnorm1(conv1)
        conv1_batchnorm_relu = self.relu(conv1_batchnorm) # (3, 14, 14)
        
        conv2 = self.conv2(conv1_batchnorm_relu)
        conv2_batchnorm = self.batchnorm2(conv2)
        conv2_batchnorm_relu = self.relu(conv2_batchnorm) # (4, 7, 7)
        
        convtranspose1 = self.convtranspose1(conv2_batchnorm_relu)
        convtranspose1_batchnorm = self.batchnorm3(convtranspose1)
        convtranspose1_batchnorm_relu = self.relu(convtranspose1_batchnorm) # (3, 14, 14)

        convtranspose2 = self.convtranspose2(convtranspose1_batchnorm_relu)
        convtranspose2_batchnorm = self.batchnorm4(convtranspose2)
        convtranspose2_batchnorm_relu = self.relu(convtranspose2_batchnorm) # (10, 28, 28)
        
        global_avg_pool = self.global_avg_pool(convtranspose2_batchnorm_relu) # (10, )
        
        global_avg_pool = global_avg_pool.view(global_avg_pool.size(0), -1)
        
        out = self.fc(global_avg_pool)
        
        return out
    
model = ConvTransposeModel().to(device=device)

summary(model, input_size=(1, 28, 28))

batch_size = 32
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
 transforms.ToTensor(),
])

train_dataset = datasets.MNIST('C:\\data/MNIST/train/',
                                 train=True,
                                 download=True,
                                 transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

valid_dataset = datasets.MNIST(root='C:\\data/MNIST/test/',
                                            train=False, 
                                            download=True,
                                            transform=transform)

valid_loader = DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


loss_dict = {}
val_loss_dict = {}
acc_dict = {}
val_acc_dict = {}
train_step = len(train_loader)
val_step = len(valid_loader)


for epoch in range(1, epochs + 1):
    loss_list = [] # losses of i'th epoch
    num_correct = 0
    num_samples = 0
    
    for train_step_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        output = model(img)
        loss = loss_fn(output, label)
        
        _, predictions = output.max(1)
        num_correct += (predictions == label).sum()
        num_samples += predictions.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if ((train_step_idx+1) % 100 == 0):
            print(f"Epoch [{epoch}/{epochs}] Step [{train_step_idx + 1}/{train_step}] Loss: {loss.item():.4f}")
            # Accuracy: {(num_correct / num_samples) * 100:.4f}

    loss_dict[epoch] = loss_list
    # acc_dict[epoch] = (num_correct / num_samples) * 100

    val_loss_list = []
    val_num_correct = 0
    val_num_samples = 0
    
    for val_step_idx, (val_img, val_label) in enumerate(valid_loader):
        with torch.no_grad():
            val_img = val_img.to(device)
            val_label = val_label.to(device)
            
            model.eval()
            val_output = model(val_img)
            val_loss = loss_fn(val_output, val_label)
            
            _, val_predictions = val_output.max(1)
            val_num_correct += (val_predictions == val_label).sum()
            val_num_samples += val_predictions.size(0)

        val_loss_list.append(val_loss.item())

    val_loss_dict[epoch] = val_loss_list
    # val_acc_dict[epoch] = (val_num_correct / val_num_samples) * 100


    print(f"Epoch [{epoch}] Train Loss: {mean(loss_dict[epoch]):.4f} Val Loss: {mean(val_loss_dict[epoch]):.4f}")
    # print(f"Epoch [{epoch}] Train Accuracy: {acc_dict[epoch]:.4f} Val Accuracy: {val_acc_dict[epoch]:.4f}")
    print("========================================================================================")

torch.save(model.state_dict(), 'model/convtranspose_model.pt')
