# CatFaceNet

- Install
  
      pip install git+https://github.com/25hanium/CatFaceNet.git

- Examle
  
      import CatFaceNet as cfn
      import numpy as np
      import torch
      from torchvision import transforms
      
      # Detector
      whichCat = cfn.CatDetector()
      # Model
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      input_shape = (3, 255, 255)
      model = cfn.getFaseNet().to(device)
      model.eval()
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      # main
      cat_img = np.ones((224, 224, 3), dtype=np.uint8)
      cat = transform(cat_img).unsqueeze(0).to(device)
      with torch.no_grad():
          embedding = model(cat)
      
      cat = whichCat(embedding)
      
      print(cat)
