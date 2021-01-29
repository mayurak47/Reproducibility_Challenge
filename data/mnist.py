import torch, torchvision
import torchvision.transforms as transforms

def mnist_dataloaders(batch_size=32):
    train_mnist_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    test_mnist_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(train_mnist_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_mnist_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
