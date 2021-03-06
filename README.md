# PyTorch_Study

[PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html)

### 외우자 이 아홉줄

```python
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        out = model(img)
        loss = loss_fn(out, label)

        optimizer.zero_grad() ## gradient initialized (어디에 있든 loss.backward() 앞에만 있으면 된다)
        loss.backward() ## back propagation (gradient(==dw(t)) updated)
        optimizer.step() ## weight updated (w(t) = w(t-1) - lr * (gradient optimized(==optimizer(dw(t))))
 ```

### Tensor shape, stride, offset
```python
# stride
>>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride()
(5, 1)

>>> x.t().stride()
(1, 5)

# offset
>>> x = torch.tensor([1, 2, 3, 4, 5])
>>> x.storage_offset()
0
>>> x[3:].storage_offset()
3
```

### L2 regularization

```python
# weight_decay -> L2 penalty (default: 0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 
```

### L1 regularization
[reference link](https://stackoverflow.com/questions/58172188/how-to-add-l1-regularization-to-pytorch-nn-model)
```python
def l1_regularizer(model, lambda_l1=0.01):
    l1_loss = 0
    for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                l1_loss += lambda_l1 * model_param_value.abs().sum()
    return l1_loss
```



### control the randomness
```python
import torch
import numpy as np
import random

random_seed = 31

random.seed(random_seed)
np.random.seed(random_seed)

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
```

### about the ConvTranspose2d
```python
import torch
import torch.nn as nn

cifar = torch.randn((1, 3, 32, 32))

"""
be careful when stride > 1
"""
out = nn.ConvTranspose2d(in_channels=3,
                         out_channels=6,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         )(cifar) ## torch.Size([1, 6, 63, 63])

print(out.size())

out2 = nn.ConvTranspose2d(in_channels=3,
                         out_channels=6,
                         kernel_size=3,
                         stride=2,
                         padding=1,
                         output_padding=1,
                         )(cifar) ## torch.Size([1, 6, 64, 64])

print(out2.size())
```

### torch static graph
```python
@torch.jit.script
```
 
### torch tensor float value count
```python
values, counts = np.unique((quantized_weight.view(-1).detach().numpy()), return_counts=True)
```

### image data pipeline && train && deploy

[link](https://urbanbase.github.io/dev/2019/12/17/Deep-Learning-Image-Classification.html)

### DataParallel

```python
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:
        net = torch.nn.DataParallel(net, list(range(gpu_num))).cuda()
        net2 = torch.nn.DataParallel(net2, list(range(gpu_num))).cuda()
```

### depthwise-seperable-convolution

```python
    def conv_dw(inp, oup, stride):
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            )
```

### 메서드 with underbar(_)
> 
> - 언더바가 있다면 : in_place (새로운 tensor가 생기지 않고 기존 tensor 변경)
>   
> - 언더바가 없다면 : 새로운 tensor를 리턴
> 
> 거의 대부분의 메소드는 언더바 버전이 있기때문에 참고하면 좋다.
> 


### contiguous() 
>
> pytorch에서는 tensor가 memory에 연속하여 올라가 있지 않으면 
>
> view(), transpose()같은 함수를 사용할 때 error가 발생한다.
>
> contiguous() -> contiguous 하게 해줌 
> 
> is_contiguous() -> contiguous 인지 확인
>

### view() VS reshape()
>
> - view() : 기존 tensor와 같은 공간 공유. 기존 tensor가 바뀔 경우 같이 바뀜.
>  
>            반대도 성립.
>
> - reshape() : 기존 tensor를 copy하고 그 tensor를 다시 view하여 반환.
>

###  transpose() & permute()
> 
> - transpose() : 2개의 차원을 변경하는데 사용
> 
> - permute() : 모든 차원의 순서를 재배치
>

### detach() VS clone() vs data
>
> - detach() : 기존 tensor에서 gradient 전파 안 되는 tensor 생성
>  
>              단, 같은 공간 공유. 이 tensor가 바뀌면 기존 tensor도 바뀜.
>
> - clone() : 기존 tensor의 내용을 copy한 tensor 생성.
>
> - data : detach() 와 똑같다. gradient계산이 잘못되어도 error 발생이 안됨. 사용하지 말자.
> 
 
### check the pointer of tensor

```python
x_ptr = x.storage().data_ptr()
y_ptr = y.storage().data_ptr()
if ( x_ptr == y_ptr):
    ...
```


### torch.nn VS torch.nn.functional
>
> layer with weights --> torch.nn
>
> layer with no weights(activation) --> torch.nn or functional ok.
> 


### model.summary()
```python
from torchsummary import summary
from torchviz import make_dot
from torch.autograd import Variable

model = Model().to(device)
summary(model, input_size=(3, 32, 32))

InTensor = Variable(torch.randn(1, 3, 32, 32)).to(device)
make_dot(model(InTensor), params=dict(model.named_parameters())).render("model", format="png")
```


### nn.CrossEntropyLoss() == nn.LogSoftmax() + nn.NLLLoss()
>
> 이 두 방식은 똑같다.
>
> 그러니까 pytorch에서는 마지막 activation 으로 **softmax를 쓸 필요 없다**
>
> nn.CrossEntropyLoss()가 다 계산해준다.
>
> legacy 코드들에 후자로 되어 있는 경우가 많은데, speed상 전자가 빠르니 사용을 권장한다고 한다.
>


### batch size가 커서 memory가 터져버릴 때
>
> 원래 batch size를 한 epoch 마다 말고
>
> 그 batch size를 1/2 혹은 1/3으로 쪼개서 2~3(혹은 그 이상) epoch마다 loss를 accumulate 하면 된다.
>


### model.eval() VS with torch.no_grad()
>
> 전자는 batchnorm과 dropout 같이 train과 eval시에 다르게 동작하는 layer에 영향을 준다.
>
> 후자는 gradient계산(computation)을 멈춘다. 즉 memory와 speed에 영향을 준다.
>
> 이것을 고려하여 사용하면 된다.
>
> 대개의 경우 훈련 시 validation loop를 만든다면 두개를 둘 다 키는 게 맞다고 본다.
>
> train loop로 다시 들어갈땐 model.train()으로 mode를 다시 바꾸는 건 잊지 말자
>


### PyTorch에서 batchnorm의 mean과 variance를 얻는 방법
>
> 100% 확신은 안 선다.
>
```python
mean_list = []
variance_list = []

for epoch in range(epochs):
    # 대충 train loop...
        model.train()
        mean = model.batchnorm.running_mean.clone() # mean을 추적한다.
        variance = model.batchnorm.running_var.clone() # variance을 추적한다.
    # 대충 evaluation loop...
        model.eval()
            mean_list.append(mean)
            variance_list.append(variance)
```

### (추가)model.pt 에서 block안에 있는 bn의 mean과 var를 얻는 법

```python
# 예를 들어 이런 block이 있으면
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(8),
        )
        
model.block1._modules['1'].running_mean.cpu().detach().numpy()
model.block1._modules['1'].running_var.cpu().detach().numpy()
model.block1._modules['4'].running_mean.cpu().detach().numpy()
model.block1._modules['4'].running_var.cpu().detach().numpy()
# 이렇게 하면 얻을 수 있다... 
```

### nn.CrossEntropyLoss() VS nn.BCEWithLogitsLoss() VS Focal Loss
>
> 1번은 multi-class 문제에 쓰인다. 한 image에 한 class만 존재하는 경우.
>
> tf.keras와 다르게 pytorch는 마지막 layer에 softmax 없이 사용해도 같이 계산해준다.
>
> 2번은 mulit-label 문제에 쓰인다. 한 image에 여러 class가 존재하는 경우.
>
> Binary Cross Entropy 라고도 하여 헷갈리기 쉽다. 조심. Sigmoid CE loss 라고도 함.
>
> 3번도 multi-label 문제에 쓰인다. [논문참고](https://arxiv.org/abs/1708.02002)
>

### unsqueeze(), squeeze(), view(), clamp()

```python
import torch
x = torch.randn(16, 3, 32, 32)
y = torch.randn(1, 3, 1, 8)

a = x.view(16, -1) ## flatten
b = y.squeeze(0) ## 차원이 하나 줄어든다.
c = y.unsqueeze(2) ## 차원이 하나 늘어난다.

p = torch.arange(-10, 11)
q = p.clamp(min=0) ## relu
```


### einsum 
```python
import torch

#sum of tensor
x = torch.randn(3, 5)

sum_of_tensor = torch.einsum("ij ->", x) # (3, 5) -> 0차원으로 만드니 더한다고 생각하면 됨.

print(sum_of_tensor)

#transpose!!
xx = torch.einsum("ij -> ji", x) # (i, j) -> (j, i) 

print(x)
print(xx)

#sum by column
sum_by_column = torch.einsum("ij -> j", x) # column만 남기므로 칼럼방향으로 더하는것

print(sum_by_column)

#sum by row
sum_by_row = torch.einsum("ij -> i", x) # 위와 동일

print(sum_by_row)

#matrix-matrix multiplication
a = torch.randn(127, 34)
b = torch.rand(13, 34)

c = torch.einsum("ij, kj -> ik", a, b) # ( i , j ) x ( k , j ) == ( i , k ) :  

print(c.shape)

#matrix-matrix element-wise multiplication
aa = torch.randn(14, 34)
bb = torch.randn(14, 34)

cc = torch.einsum("ij, ij -> ij", aa, bb)

print(cc.shape)

#dot product
aa = torch.randn(14, 34)
bb = torch.randn(14, 34)

cc = torch.einsum("ij, ij -> ", aa, bb)

print(cc.shape)

#batch matrix multiplication
p = torch.randn(3, 2, 5)
q = torch.randn(3, 5, 3)

r = torch.einsum("nij, njk -> nik", p, q)

print(r.shape)
```
