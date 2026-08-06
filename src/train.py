import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from model import BrainTumorCNN

model = BrainTumorCNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

DATASET_DIR = Path("dataset")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):

    running_loss = 0.0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    MODELS_DIR / "brain_tumor_cnn.pth"
)

