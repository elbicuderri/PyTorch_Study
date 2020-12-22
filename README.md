# PyTorch_Study


[LSTM visualization](https://www.youtube.com/watch?v=8HyCNIVRbSU&feature=youtu.be)

[ResNet visualization](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)


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

**einsum is all you need**
> 꼭 써라 두 번 써라
>
> 바로 예제 코드를 보자

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
