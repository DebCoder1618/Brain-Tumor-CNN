import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from model import BrainTumorCNN

DATASET_DIR = Path("dataset")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

model = BrainTumorCNN()

MODELS_DIR = Path("models")

model.load_state_dict(torch.load(MODELS_DIR / "brain_tumor_cnn.pth"))

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")