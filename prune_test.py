import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.prune as prune

from torchsummary import summary

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):
    """
    Toy neural network
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(epoch, network, train_loader, train_losses, train_counter, optimizer):
    # specify you will be training
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), '/home/ubuntu/code/mnist/results/model.pth')
        torch.save(optimizer.state_dict(), '/home/ubuntu/code/mnist/results/optimizer.pth')

def train_loop():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/home/ubuntu/code/mnist/images', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/home/ubuntu/code/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, train_losses, train_counter, optimizer)
        test(network, test_loader, test_losses)


def compute_num_parameters(network):
    num_weights = 0
    num_biases = 0
    for cnt, module in enumerate(network.modules()):
        if cnt == 0:
            continue
        param_list = list(module.named_parameters())
        # nothing
        if len(param_list) == 0:
            continue
        # weight and biases
        if len(param_list) == 2:
            if param_list[0][0] == 'weight':
                num_weights = num_weights + np.prod(param_list[0][1].shape)
            else:
                num_biases = num_biases + np.prod(param_list[0][1].shape)

            if param_list[1][0] == 'weight':
                num_weights = num_weights + np.prod(param_list[1][1].shape)
            else:
                num_biases = num_biases + np.prod(param_list[1][1].shape)

    return num_weights, num_biases


def prune_tutorial():
    network = Net()
    network.load_state_dict(torch.load('/home/ubuntu/code/mnist/results/model.pth'))

    modules = [module for module in network.modules()]
    print(modules[1:])
    print(' ----------------')
    # Print summary
    # device = 'cuda'
    # summary(network.to(device), (1, 28, 28))
    num_weights, num_biases = compute_num_parameters(network)

    for cnt, module in enumerate(network.modules()):
        if cnt != 1:
            continue
        prune.random_unstructured(module, name="weight", amount=0.3)
        prune.l1_unstructured(module, name="bias", amount=3)
        # The below makes pruning permanent
        prune.remove(module, 'weight')
        prune.remove(module, 'bias')
        # print(list(module.named_parameters()))
        # print(list(module.named_buffers()))

    # pass
    for cnt, module in enumerate(network.modules()):
        if cnt != 1:
            continue
        print(list(module.named_parameters()))
    torch.save(network.state_dict(), '/home/ubuntu/code/mnist/results/model_pr.pth')
    torch.save(network, '/home/ubuntu/code/mnist/results/model_pr_full.pth')

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/home/ubuntu/code/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    test_losses = []
    test(network, test_loader, test_losses)

    pass

if __name__ == '__main__':
    print('Pruning test')
    # train_loop()
    prune_tutorial()


