import torch
import torch.nn as nn

x = torch.randn((2, 4), requires_grad=True)
w = torch.randn((4, 10), requires_grad=True)

target = torch.empty(2, dtype=torch.long).random_(10)

print(target)

y = torch.mm(x, w)

print(y)

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(y, target)

print(loss)

loss.backward()

# print(y)

# print(y.retain_grad())

print(x.grad)

print(w.grad)

# y = torch.argmax(y, dim=1)

# print(y)

