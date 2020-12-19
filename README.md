# PyTorch_Study


[LSTM 설명](https://www.notion.so/LSTM-f0f02751efa641d09b1ab44c94aa7528#880ef9f7cf79481faf538a8a7ac7ae9c)

[ResNet 시각화](https://www.notion.so/ResNet-13c65d4ff0a541448acce69d4454c86f#0f755b44ea9646f080f38f039d33e372)


**model.eval() Vs with torch.no_grad()**

전자는 batchnorm과 dropout에 영향을 준다.

후자는 gradient계산(computation)을 멈춘다. 즉 memory와 speed에 영향을 준다.

이것을 고려하여 사용하면 될 것이다.

