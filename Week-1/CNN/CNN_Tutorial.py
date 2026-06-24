import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):

        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))

        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)

        s4 = torch.flatten(s4, 1)

        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))

        output = self.fc3(f6)

        return output


# Create network
net = Net()

print(net)

# Network parameters
params = list(net.parameters())

print(len(params))
print(params[0].size())

# Create dummy input
input = torch.randn(1, 1, 32, 32)

# Forward pass
out = net(input)

print(out)

# Backward with random gradient
net.zero_grad()
out.backward(torch.randn(1, 10))

# Create output again
output = net(input)

# Dummy target
target = torch.randn(10)
target = target.view(1, -1)

# Loss function
criterion = nn.MSELoss()

# Compute loss
loss = criterion(output, target)

print(loss)

# Inspect computation graph
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# Clear gradients
net.zero_grad()

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

# Backpropagation
loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# ==========================
# Optimizer Section
# ==========================

import torch.optim as optim

# Create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training step
optimizer.zero_grad()

output = net(input)

loss = criterion(output, target)

loss.backward()

optimizer.step()

print("Weights updated successfully!")
