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

#matrix diagonal
g = torch.rand(2, 2)
print(g)

gg = torch.einsum("ii->i", g)
print(gg)

#matrix trace
ggg = torch.einsum("ii->", g)
print(g)

# #Fully Connected Layer 
a = torch.randn(32, 3, 228, 228)

b = torch.randn(32, 228, 228, 3)

w1 = torch.randn(10, 3 * 228 * 228)

w2 = torch.randn(228 * 228 * 3, 10)

y1 = torch.einsum("nchw, kchw-> nk", a, w1.reshape(10, 3, 228, 228)) #PyTorch

y2 = torch.einsum("nhwc, hwck-> nk", b, w2.reshape(228, 228, 3, 10)) #TensorFlow

print(y1.size())

print(y2.size())