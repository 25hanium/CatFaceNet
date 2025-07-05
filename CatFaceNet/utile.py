from .model import getFaseNet
from importlib.resources import files
import numpy as np
import torch
import torch.nn.functional as F

profile_default_path = files('CatFaceNet').joinpath('profile.npy')

class CatDetector():
    def __init__(self, profile_path=profile_default_path):
        self.profile = torch.from_numpy(np.load(profile_path))


    def __call__(self, embedding):
        similarity = F.cosine_similarity(embedding, self.profile, dim=1)
        best = int(torch.argmax(similarity).cpu().item())

        return best