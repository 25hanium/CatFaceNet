from .model import getFaseNet
from importlib.resources import files
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

profile_default_path = files('CatFaceNet').joinpath('profile.npy')

class CatDetector():
    def __init__(self, profile_path=profile_default_path):
        self.profile = torch.from_numpy(np.load(profile_path))
        # Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = (3, 255, 255)
        self.model = getFaseNet().to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, cat):
        cat = self.transform(cat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(cat)

        similarity = F.cosine_similarity(embedding, self.profile, dim=1)
        best = int(torch.argmax(similarity).cpu().item())

        return best