# This is the most naive iteration.
# Next, do it with Horovod and then with PyTorch DDP.
# Also, try to do it with some other strategies: async gradients, model parallel, tensor parallel, pipeline parallel, etc.

from mpi4py import MPI
import torch 
import torch.nn as nn
from model import CNN 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device("cuda:{0}".format(rank))

# instantiate the model; broadcast to other ranks if rank == 0
# if rank > 0, receive the model from rank 0
def instantiate_model(rank):
    if rank == 0:
        model = CNN().to(device)
        comm.bcast(model, root=0)
    else:
        model = comm.bcast(None, root=0).to(device)
    # print the first few model parameters to make sure they're the same 
    print("Rank {0} model parameters:".format(rank))
    for param in model.parameters():
        print(param[0][0][0][0])
        break

    return model 

model = instantiate_model(rank)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
