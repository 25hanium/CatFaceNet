# CatFaceNet

- Install
  
      pip install git+https://github.com/25hanium/CatFaceNet

- Examle
  
      import CatFaceNet as cfn
      import numpy as np
      
      whichCat = cfn.CatDetector()
      
      cat_img = np.ones((224, 224, 3), dtype=np.uint8)
      cat = whichCat(cat_img)
      
      print(cat)
