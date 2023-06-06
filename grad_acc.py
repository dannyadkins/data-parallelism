# This is the most naive iteration.
# Next, do it with Horovod and then with PyTorch DDP.
# Also, try to do it with some other strategies: async gradients, model parallel, tensor parallel, pipeline parallel, etc.
# Also, try proper sharded dataloader, memmap... 
# Also try ring-allreduce, hierarchical allreduce, etc.

from mpi4py import MPI
import torch 
import torch.nn as nn
from model import CNN 
from utils import init_mpi, init_model, get_full_dataset, get_shard, plot_losses, fwd, train_model, test_model
import numpy as np 
import matplotlib.pyplot as plt 

comm, rank, size, device = init_mpi()

torch.manual_seed(0)

full_dataset = get_full_dataset()
images, labels = get_shard(full_dataset, rank, size)
images = images.to(device)
labels = labels.to(device)


# train the model for 10 epochs
batch_size = 256
num_epochs = 5

# def update_func(comm, rank, size, device, model, optim, step):

gradcounts = []
times = []
accs = []

# train_model(comm, rank, size, device, model, optim, loss_fn, images, labels, update_func=update_func, num_epochs=100, show_logs=True, training_curve=training_curve)
for grads_per_update in range(1, 20, 1):
    # start time 
    start_time = MPI.Wtime()

    model = init_model(comm, rank, device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    training_curve = []

    if (rank == 0):
        print("Running for ", grads_per_update, " grads per update")
    for epoch in range(num_epochs):
        
        grads = [] 
        for param in model.parameters():
            grads.append(0)

        for i in range(len(images)//batch_size):
            mini = slice(i*batch_size, (i+1)*batch_size)

            loss = fwd(model, optim, loss_fn, images[mini], labels[mini])
            loss.backward()

            # add the gradients to grads list to create a sum 
            j = 0
            for param in model.parameters():
                grads[j] = grads[j] + param.grad.data.cpu().numpy()
                j += 1

            if (i % grads_per_update == 0):
                # calculate the gradients and allreduce 
                j = 0
                for param in model.parameters():
                    grad = grads[j]
                    comm.Allreduce(MPI.IN_PLACE, grad, op=MPI.SUM)  # in-place sum of gradients across all GPUs
                    avg_grad = torch.from_numpy(grad / grads_per_update / size ).to(device)  # convert back to tensor and average
                    # update with the average gradient
                    param.grad.data = avg_grad
                    j += 1

                grads = []
                for param in model.parameters():
                    grads.append(0)

                
                # update the model parameters
                optim.step()
            
            # print("Rank {0} epoch {1} batch {2} loss {3}".format(rank, epoch, i, loss.item()))

            training_curve.append(loss.item())
    
    end_time = MPI.Wtime()

    acc = test_model(model, device)

    # print the first few model parameters to make sure they're the same
    # print("Rank {0} model parameters:".format(rank))
    # for param in model.parameters():
    #     print(param[0][0][0][0])
    #     break
    

    gradcounts.append(grads_per_update)
    times.append(end_time - start_time)
    accs.append(acc.cpu().numpy())

    print("Time taken: ", end_time - start_time)
    print("Accuracy: ", acc)

    # plot the training curve
    plot_losses(training_curve, "plots/dp_rank{0}_size{1}_ga{2}".format(rank,size,grads_per_update))

# plot gradcounts, times, accs:
if (rank == 0):
    plt.plot(gradcounts, times)
    plt.xlabel("Grads per update")
    plt.ylabel("Time taken")
    plt.savefig("plots/dp_rank{0}_size{1}_time".format(rank,size))
    plt.clf()

    plt.plot(gradcounts, accs)
    plt.xlabel("Grads per update")
    plt.ylabel("Accuracy")
    plt.savefig("plots/dp_rank{0}_size{1}_acc".format(rank,size))
    plt.clf()
