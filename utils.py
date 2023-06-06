import torch 
from model import CNN 

# instantiate the model; broadcast to other ranks if rank == 0
# if rank > 0, receive the model from rank 0
def instantiate_model(comm, rank, device):
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