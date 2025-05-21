import torch
from torchvision import transforms
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from training.model import SketchNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = SketchNet()
model.load_state_dict(torch.load("Models/sketchify_cnn.pt", map_location=device, weights_only=True))
model.to(device)

model.eval()

# Transfrom the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_sketch(image: Image.Image) -> Image.Image:
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)[0].cpu()
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image