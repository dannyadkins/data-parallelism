# This is the most naive iteration.
# Next, do it with Horovod and then with PyTorch DDP.
# Also, try to do it with some other strategies: async gradients, model parallel, tensor parallel, pipeline parallel, etc.
# Also, try proper sharded dataloader, memmap... 
# Also try ring-allreduce, hierarchical allreduce, etc.

from mpi4py import MPI
import torch 
import torch.nn as nn
from model import CNN 
from utils import init_mpi, init_model, get_full_dataset, get_shard, plot_losses, fwd, train_model

comm, rank, size, device = init_mpi()

torch.manual_seed(0)

model = init_model(comm, rank, device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

full_dataset = get_full_dataset()
images, labels = get_shard(full_dataset, rank, size)
images = images.to(device)
labels = labels.to(device)

training_curve = []

# train the model for 10 epochs
batch_size = 256

def update_func(comm, rank, size, device, model, optim):
    for param in model.parameters():
        grad_numpy = param.grad.data.cpu().numpy()  # convert tensor to numpy array
        comm.Allreduce(MPI.IN_PLACE, grad_numpy, op=MPI.SUM)  # in-place sum of gradients across all GPUs
        avg_grad = torch.from_numpy(grad_numpy / size).to(device)  # convert back to tensor and average
        # update with the average gradient
        param.grad.data = avg_grad
    
    # update the model parameters
    optim.step()

train_model(comm, rank, size, device, model, optim, loss_fn, images, labels, update_func=update_func, num_epochs=100, show_logs=True, training_curve=training_curve)

# plot the training curve
plot_losses(training_curve, "dp_rank{0}_size{1}.png".format(rank,size))

if rank == 0:
    torch.save(model.state_dict(), "dp_rank{0}_size{1}.pt".format(rank,size))