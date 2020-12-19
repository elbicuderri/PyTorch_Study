# PyTorch_Study


[LSTM visualization](https://www.youtube.com/watch?v=8HyCNIVRbSU&feature=youtu.be)

[ResNet visualization](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)


**model.eval() Vs with torch.no_grad()**

> 전자는 batchnorm과 dropout에 영향을 준다.
>
> 후자는 gradient계산(computation)을 멈춘다. 즉 memory와 speed에 영향을 준다.
>
> 이것을 고려하여 사용하면 될 것이다.

