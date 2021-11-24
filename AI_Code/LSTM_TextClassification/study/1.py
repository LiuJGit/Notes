import torch

a = torch.ones([1,3,4])

b = torch.zeros([1,3,4])

c = torch.cat([a,b],dim=1)
print(a.shape)
print(b.shape)
print(c.shape)