import torch
import argparse
import os
from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel

N, D_in, D_out = 64, 1024, 16
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
z = model(x)
import pdb; pdb.set_trace()
print(z)
