import torch
from nets import *

def get_optimizer(model, lr, momentum):
    if isinstance(model,CNN_MNIST):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif isinstance(model,CNN_CIFFAR10):
        return torch.optim.Adam(model.parameters())
        # return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(net, trainloader, optimizer, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)
    net.train()
    for batch in trainloader:
        if isinstance(net, CNN_MNIST):
            key_name = "image"
        elif isinstance(net, CNN_CIFFAR10):
            key_name = "img"
        images, labels = batch[key_name].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            if isinstance(net, CNN_MNIST):
                key_name = "image"
            elif isinstance(net, CNN_CIFFAR10):
                key_name = "img"
            images, labels = batch[key_name].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    loss_ = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return loss_, accuracy
