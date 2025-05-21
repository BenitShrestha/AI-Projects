from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import os

class SketchifyDataset(Dataset):
    def __init__(self, image_dir, image_size=(256, 256)):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size)  # resize input image

        # Simulate sketch with edge detection
        sketch = image.convert("L").filter(ImageFilter.FIND_EDGES)
        sketch = sketch.resize(self.image_size)  # resize sketch too

        image_tensor = self.transform(image)
        sketch_tensor = self.transform(sketch)

        return image_tensor, sketch_tensor