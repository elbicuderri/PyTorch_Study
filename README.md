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

#Fully Connected Layer 
a = torch.randn(32, 3, 228, 228)

b = torch.randn(32, 228, 228, 3)

w1 = torch.randn(10, 3 * 228 * 228)

w2 = torch.randn(228 * 228 * 3, 10)

y1 = torch.einsum("nchw, kchw-> nk", a, w1.reshape(10, 3, 228, 228)) #PyTorch

y2 = torch.einsum("nhwc, hwck-> nk", b, w2.reshape(228, 228, 3, 10)) #TensorFlow

print(y1.size())

print(y2.size())
```
