import torch

#z = torch.Tensor([[1,2,3,4], [5,6,7,8]])

#print(z)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

from UNet import u_net
a = u_net.UNet().to(device=device)
print(a)