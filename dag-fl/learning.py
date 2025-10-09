import torch
from nets import *
from collections import OrderedDict

def get_optimizer(net):
    if isinstance(net,MLP_MNIST):
        return torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # if isinstance(net,CNN_MNIST):
    #     return torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    elif isinstance(net,CNN_CIFFAR10):
        return torch.optim.Adam(net.parameters())
        # return torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # has overfit problem after about 50 epochs

def train(net, trainloader, optimizer, local_epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)
    net.train()
    for epoch in range(local_epochs):
        for batch in trainloader:
            if isinstance(net, MLP_MNIST):
            # if isinstance(net, CNN_MNIST):
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
            if isinstance(net, MLP_MNIST):
            # if isinstance(net, CNN_MNIST):
                key_name = "image"
            elif isinstance(net, CNN_CIFFAR10):
                key_name = "img"
            images, labels = batch[key_name].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    losses = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return losses, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

