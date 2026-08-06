import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("~/Downloads/brain_tumor_dataset").expanduser()

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
TEST_DIR = PROJECT_DIR / "dataset" / "test"

yes_images = list((SOURCE_DIR / "yes").iterdir())
no_images = list((SOURCE_DIR / "no").iterdir())

print(f"Yes images: {len(yes_images)}")
print(f"No images: {len(no_images)}")

random.shuffle(yes_images)
random.shuffle(no_images)

yes_split = int(len(yes_images) * 0.8)
no_split = int(len(no_images) * 0.8)

print(f"Yes training images: {yes_split}")
print(f"Yes testing images: {len(yes_images) - yes_split}")

print(f"No training images: {no_split}")
print(f"No testing images: {len(no_images) - no_split}")

# Create the folders
(TRAIN_DIR / "yes").mkdir(parents=True, exist_ok=True)
(TRAIN_DIR / "no").mkdir(parents=True, exist_ok=True)

(TEST_DIR / "yes").mkdir(parents=True, exist_ok=True)
(TEST_DIR / "no").mkdir(parents=True, exist_ok=True)

print("Folders created successfully!")

for image in yes_images[:yes_split]:
    shutil.copy(image, TRAIN_DIR / "yes")

for image in yes_images[yes_split:]:
    shutil.copy(image, TEST_DIR / "yes")

for image in no_images[:no_split]:
    shutil.copy(image, TRAIN_DIR / "no")

for image in no_images[no_split:]:
    shutil.copy(image, TEST_DIR / "no")

print("Dataset split completed successfully!")