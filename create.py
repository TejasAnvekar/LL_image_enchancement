import cv2
import os
import numpy as np
from tqdm import tqdm

outpath = "/home/tejas/Desktop/low_light/validation/archi/"
inpath = "/media/tejas/TAS/arch_data/validation/architecture"




for i in tqdm(os.listdir(inpath)):
    try:
        img = cv2.imread(os.path.join(inpath,i))
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        prob = np.random.uniform(low=0.85,high=1.0,size=1)
        hsv[...,2] = hsv[...,2]*(1-prob)
        cv2.imwrite(os.path.join(outpath,i),cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR))
    except:
        print(i,"error image")
        os.remove(os.path.join(inpath,i))
        pass


