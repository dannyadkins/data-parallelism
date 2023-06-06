# This is the most naive iteration.
# Next, do it with Horovod and then with PyTorch DDP.
# Also, try to do it with some other strategies: async gradients, model parallel, tensor parallel, pipeline parallel, etc.
# Also, try proper sharded dataloader, memmap... 
# Also try ring-allreduce, hierarchical allreduce, etc.

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

torch.manual_seed(0)

model = instantiate_model(rank)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

from datasets import load_dataset 
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def get_data():
    dataset = load_dataset('mnist', split='train')
    return dataset['image'], dataset['label']

def transforms(examples):
    # each example is a PIL.PngImagePlugin.PngImageFile
    # resize to 28x28
    # convert to tensor
    to_tensor = ToTensor()
    # compose
    transforms = Compose([to_tensor])
    examples = [transforms(example) for example in examples]
    examples = torch.stack(examples)
    return examples

def preprocess_data(images, labels):
    # each image is a PIL.PngImagePlugin.PngImageFile

    # open it and turn it into a 2d tensor
    images = transforms(images)
    
    # one hot encode the labels
    # labels = torch.nn.functional.one_hot(labels, num_classes=10)
    labels = torch.tensor(labels)

    return images, labels

raw_images, raw_labels = get_data()[:20000]
full_dataset = preprocess_data(raw_images, raw_labels)

# get the shard of the data corresponding to this rank
def get_shard(data, rank, size):
    # data is a tuple of (images, labels)
    images, labels = data
    # get the shard size
    shard_size = len(images) // size
    # get the start and end indices of the shard
    start = rank * shard_size
    end = start + shard_size
    # get the shard
    images = images[start:end]
    labels = labels[start:end]
    return images, labels

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