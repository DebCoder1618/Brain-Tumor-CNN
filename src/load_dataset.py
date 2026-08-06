from torchvision import datasets, transforms
from pathlib import Path
from torch.utils.data import DataLoader

DATASET_DIR = Path("dataset")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root=DATASET_DIR / "train",
    transform=transform
)

print("Number of training images:", len(train_dataset))
print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)

image, label = train_dataset[0]

print(type(image))
print(image.shape)
print(label)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("\nBatch shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)