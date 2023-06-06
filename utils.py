import torch 
from model import CNN 
import matplotlib.pyplot as plt 
from mpi4py import MPI

def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    device = torch.device("cuda:{0}".format(rank))
    return comm, rank, size, device

# instantiate the model; broadcast to other ranks if rank == 0
# if rank > 0, receive the model from rank 0
def init_model(comm, rank, device):
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


from datasets import load_dataset 
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def get_raw_data():
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

def get_full_dataset():
    raw_images, raw_labels = get_raw_data()
    return preprocess_data(raw_images, raw_labels)

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

def plot_losses(losses, name):
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # save as name.png
    plt.savefig(name + ".png")
    plt.close()

# some function to do a pass without updating the model
def fwd(model, optim, loss_fn, images, labels):
    # zero the gradients
    optim.zero_grad()
    # forward pass
    output = model(images)
    # compute loss
    loss = loss_fn(output, labels)
    # backward pass
    return loss

def train_model(comm, rank, size, device, model, optim, loss_fn, images, labels, update_func, num_epochs=1, batch_size=256, training_curve=[], show_logs=False):
    for epoch in range(num_epochs):
        for i in range(len(images)//batch_size):
            mini = slice(i*batch_size, (i+1)*batch_size)

            loss = fwd(model, optim, loss_fn, images[mini], labels[mini])
            loss.backward()
            if (show_logs):
                print("Rank {0} epoch {1} loss: {2}".format(rank, epoch, loss.item()))
            # calculate the gradients and allreduce 
            update_func(comm, rank, size, device, model, optim)

            training_curve.append(loss.item())

    # print the first few model parameters to make sure they're the same
    if (show_logs):
        print("Rank {0} model parameters:".format(rank))
        for param in model.parameters():
            print(param[0][0][0][0])
            break