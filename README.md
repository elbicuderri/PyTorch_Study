# PyTorch_Study


### 외우자 이 아홉줄

```python
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        out = model(img)
        loss = loss_fn(out, label)

        optimizer.zero_grad() ## gradient initialized (어디에 있든 loss.backward() 앞에만 있으면 된다)
        loss.backward() ## back propagation (gradient(=dw(t) updated)
        optimizer.step() ## weight updated (w(t) = w(t-1) - lr * (gradient optimized(=optimizer(dw(t))))
 ```
 
### contiguous() 란
>
> pytorch에서는 tensor가 memory에 연속하여 올라가 있지 않으면 
>
> view(), transpose()같은 함수를 사용할 때 error가 발생한다.
>
> contiguous(), is_contiguous()
>

### view() VS reshape()
>
> - view() : 기존 tensor와 같은 공간 공유. 기존 tensor가 바뀔 경우 같이 바뀜.
>  
>            반대도 성립.
>
> - reshape() : 기존 tensor를 copy하고 그 tensor를 다시 view하여 반환.
>


### detach() VS clone() 
>
> - detach() : 기존 tensor에서 gradient 전파 안 되는 tensor 생성
>  
>              단, 같은 공간 공유. 이 tensor가 바뀌면 기존 tensor도 바뀜.
>
> - clone() : 기존 tensor의 내용을 copy한 tensor 생성.
>

 
### tesnsor의 메모리 주소 확인

```python
x_ptr = x.storage().data_ptr()
y_ptr = y.storage().data_ptr()
if ( x_ptr == y_ptr):
    ...
```


### torch.nn VS torch.nn.functional
>
> **결론만 말하면 nn으로 통일해서 쓰자**
> 


### model.summary() 하는 법
```python
from torchsummary import summary
from torchviz import make_dot
from torch.autograd import Variable

model = SimpleResNet().to(device)
summary(model, input_size=(3, 32, 32))

InTensor = Variable(torch.randn(1, 3, 32, 32)).to(device)
make_dot(model(InTensor), params=dict(model.named_parameters())).render("model", format="png")
```


### nn.CrossEntropyLoss() == nn.LogSoftmax() + nn.NLLLoss()
>
> 이 두 방식은 똑같다.
>
> 그러니까 pytorch에서는 마지막 activation 으로 **softmax를 쓸 필요 없다!!!**
>
> nn.CrossEntropyLoss()가 다 계산해준다!!
>
> 계산을 어떻게 해주는 지는 알아서 찾아보시길.. (계산 식 쓰면 머리만 아프니...)
>
> 특히 옛날 코드들에 후자로 되어 있는 경우가 많은데, speed상 전자가 빠르니 사용을 권장한다고 한다.
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


### PyTorch에서 batchnorm의 mean과 variance를 얻기가 은근 어렵다
>
> 일단 찾아낸 방법...
>
> 정확하지 않은데... 약간 더 세밀한 debugging이 필요하다... 그리고 batchnorm 방식이 약간씩 달라서리...
>

```python
mean_list = []
variance_list = []

for epoch in range(epochs):
    # 대충 train loop...
        model.train()
        mean = model.batchnorm.running_mean.clone()
        variance = model.batchnorm.running_var.clone()
    # 대충 evaluation loop...
        model.eval()
            mean_list.append(mean)
            variance_list.append(variance)
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

a = x.view(16, -1)
b = y.squeeze(0)
c = y.unsqueeze(2)

p = torch.arange(-10, 11)
q = p.clamp(min=0)
```


### einsum is all you need
> 꼭 써라 두 번 써라
>
> 바로 예제 코드를 보자
>
```python
import torch

#sum of tensor
x = torch.randn(3, 5)

sum_of_tensor = torch.einsum("ij ->", x)

print(sum_of_tensor)

#transpose!!
xx = torch.einsum("ij -> ji", x)

print(x)
print(xx)

#sum by column
sum_by_column = torch.einsum("ij -> j", x)

print(sum_by_column)

#sum by row
sum_by_row = torch.einsum("ij -> i", x)

print(sum_by_row)

#matrix-matrix multiplication
a = torch.randn(127, 34)
b = torch.rand(13, 34)

c = torch.einsum("ij, kj -> ik", a, b)

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
~~이거 뭐 사기 아니냐~~

[docker study용 github](https://github.com/open-mmlab/mmdetection)
