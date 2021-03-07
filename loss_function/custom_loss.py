import torch
import torch.nn as nn
import torch.nn.functional as F

def my_cross_entropy(x, y):
    log_prob = -1.0 * F.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    return loss

x = torch.randn((2, 4), requires_grad=True)
w = torch.randn((4, 10), requires_grad=True)

target = torch.empty(2, dtype=torch.long).random_(10)

print(target)

y = torch.mm(x, w)

loss = my_cross_entropy(y, target)
print(loss)

loss_fn = nn.CrossEntropyLoss()
loss_2 = loss_fn(y, target)
print(loss_2)

# loss.backward()

# print(x.grad)

# print(w.grad)

# loss_2.backward()

# print(x.grad)

# print(w.grad)