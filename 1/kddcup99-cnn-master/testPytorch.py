import torch
# 以下代码只有在PyTorch GPU版本上才会执行
import time

print(torch.__version__)
print(torch.cuda.is_available())
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
