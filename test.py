import os
import glob
from ultralytics import YOLO

test_imgs = glob.glob(os.path.join('./test/images')+'\*.jpg')

model = YOLO('./best.pt')

for img in test_imgs:
    model(img, save=True)