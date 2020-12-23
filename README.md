# PyTorch_Study

**외우자 이 아홉줄**

```python
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        
        model.train()
        out = model(img)
        loss = loss_fn(out, label)

        optimizer.zero_grad() ## gradient initialized
        loss.backward() ## back propagation ( gradient updated )
        optimizer.step() ## weight updated ( w(t) = w(t-1) - lr * gradient )
 ```
 
**nn.CategoricalCrossentropy() == nn.LogSoftmax() + nn.nLLLos()**
>
> 이 두 방식은 똑같다.
>
> 그러니까 pytorch에서는 마지막 activation 으로 **softmax를 쓸 필요 없다!!!**
>
> nn.CategoricalCrossentropy() 이 다 계산해준다!!
>
> 계산을 어떻게 해주는 지는 알아서 찾아보시길..
>
> 특히 옛날 코드들에 후자로 되어 있는 경우가 많은데, speed상 전자가 빠르 사용을 권장한다고 한다.
>

**batch size가 커서 memory가 터져버릴 때**
>
> 한 epoch말고
>
> 두~세 epoch마다 loss를 accumulate 하면 된다. code로 구현은 시간나면^^
>


**model.eval() Vs with torch.no_grad()**

> 전자는 batchnorm과 dropout 같이 train과 eval시에 다르게 동작하는 layer에 영향을 준다.
>
> 후자는 gradient계산(computation)을 멈춘다. 즉 memory와 speed에 영향을 준다.
>
> 이것을 고려하여 사용하면 된다.


**nn.CrossEntropyLoss() Vs nn.BCEWithLogitsLoss() Vs Focal Loss**

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


**einsum is all you need**
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
