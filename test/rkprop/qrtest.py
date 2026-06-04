import numpy as np
import torch
import torch.linalg as tla

torch.set_default_dtype(torch.float64)
n = 10
A = torch.randn((10,10))
B = torch.zeros((10,10))
AB = torch.cat((A,B),dim=1).T
BA = torch.cat((B,A),dim=1).T
print("AB shape = ",AB.shape)
print("BA shape = ",BA.shape)

Qab,Rab = tla.qr(AB)
Qba,Rba = tla.qr(BA)
Qa,Ra = tla.qr(A)

print(" Qba = ",Qba)

print("errAB = ",tla.norm(Qab@Rab-AB)/tla.norm(AB))
print("errBA = ",tla.norm(Qba@Rba-BA)/tla.norm(BA))
print("errA = ",tla.norm(Qa@Ra-A)/tla.norm(A))