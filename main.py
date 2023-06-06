# This is the most naive iteration.
# Next, do it with Horovod and then with PyTorch DDP.
# Also, try to do it with some other strategies: async gradients, model parallel, tensor parallel, pipeline parallel, etc.
# Also, try proper sharded dataloader, memmap... 
# Also try ring-allreduce, hierarchical allreduce, etc.

from mpi4py import MPI
import torch 
import torch.nn as nn
from model import CNN 
from utils import instantiate_model, get_full_dataset, get_shard

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device("cuda:{0}".format(rank))

torch.manual_seed(0)

model = instantiate_model(comm, rank, device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

full_dataset = get_full_dataset()
images, labels = get_shard(full_dataset, rank, size)
images = images.to(device)
labels = labels.to(device)

# train the model
def train(model, optim, loss_fn, images, labels):
    # zero the gradients
    optim.zero_grad()
    # forward pass
    output = model(images)
    # compute loss
    loss = loss_fn(output, labels)
    # backward pass
    loss.backward()
    return loss.item()

training_curve = []

# train the model for 10 epochs
batch_size = 256
for epoch in range(5):
    for i in range(len(images)//batch_size):
        mini = slice(i*batch_size, (i+1)*batch_size)
        loss = train(model, optim, loss_fn, images[mini], labels[mini])

        print("Rank {0} epoch {1} loss: {2}".format(rank, epoch, loss))
        # calculate the gradients and allreduce 
        for param in model.parameters():
            grad_numpy = param.grad.data.cpu().numpy()  # convert tensor to numpy array
            comm.Allreduce(MPI.IN_PLACE, grad_numpy, op=MPI.SUM)  # in-place sum of gradients across all GPUs
            avg_grad = torch.from_numpy(grad_numpy / size).to(device)  # convert back to tensor and average
            # update with the average gradient
            param.grad.data = avg_grad
        
        # update the model parameters
        optim.step()
        
        training_curve.append(loss)

    # print the first few model parameters to make sure they're the same
    print("Rank {0} model parameters:".format(rank))
    for param in model.parameters():
        print(param[0][0][0][0])
        break

# plot the training curve
import matplotlib.pyplot as plt
plt.plot(training_curve)
plt.show()
# save the figure to a local file ./training_curve_{rank}.png
plt.savefig("training_curve_rank{0}_size{1}.png".format(rank, size))

if rank == 0:
    torch.save(model.state_dict(), "model.pt")