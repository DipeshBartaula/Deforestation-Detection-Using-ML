import os
# torch
import torch
from fastai.vision.all import *
from PIL import Image
from fastai.vision.core import *
import torchvision.transforms as transforms


class ImageClassifier:
    def __init__(self):
        self.classes = ['AnnualCrop',
                        'Forest',
                        'HerbaceousVegetation',
                        'Highway',
                        'Industrial',
                        'Pasture',
                        'PermanentCrop',
                        'Residential',
                        'River',
                        'SeaLake']
        self.transformer = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def model_predict(self, img_path, model):
        # Load the image
        img = Image.open(img_path).convert('RGB')
        img = self.transformer(img).unsqueeze(0)

        # Use the model to make a prediction
        with torch.no_grad():
            output = model(img)
        preds = torch.softmax(output, dim=1)

        return preds.argmax(dim=1)

    def allowed_file(self, filename, allowed_extensions):
        return '.' in filename and \
               filename.rsplit('.', 1)[1] in allowed_extensions
