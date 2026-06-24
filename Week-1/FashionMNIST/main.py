import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Download training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),
)

# Download test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),
)

batch_size = 128

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Show data shape
for X, y in test_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)

            print(
                f"loss: {loss_value:>7f} [{current:>5d}/{size:>5d}]"
            )


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (
                (pred.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    test_loss /= num_batches

    correct /= size

    print(
        f"Accuracy: {(100 * correct):>0.1f}%"
        f", Avg loss: {test_loss:>8f}"
    )


epochs = 10

for t in range(epochs):

    print(f"\nEpoch {t + 1}")
    print("---------------------")

    train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    test(
        test_dataloader,
        model,
        loss_fn
    )

print("Done!")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Load model
loaded_model = NeuralNetwork().to(device)

loaded_model.load_state_dict(
    torch.load("model.pth", weights_only=True)
)

print("Loaded model successfully")

# FashionMNIST classes
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Predict one test image
loaded_model.eval()

image, label = test_data[0]

with torch.no_grad():

    image = image.unsqueeze(0).to(device)

    pred = loaded_model(image)

    predicted = classes[pred.argmax(1).item()]
    actual = classes[label]

    print(f'Predicted: "{predicted}"')
    print(f'Actual: "{actual}"')