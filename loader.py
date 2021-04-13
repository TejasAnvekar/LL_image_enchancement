from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random

random.seed(1)




def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.0001
        sigma = var**0.05
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy =  gauss + image
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 1.0
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
            for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i , int(num_pepper))
            for i in image.shape]
        out[coords] = 0

        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy



def adjust_gamma(image, gamma=1.0):
#    image = noisy(image,"s&p")
    
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)



class lowlightloader(Dataset):
    def __init__(self,root,transforms_img=None,transforms_freq=None):
        super(lowlightloader,self).__init__()
        self.root = root
        self.transforms_img = transforms_img
        # self.transforms_freq = transforms_freq
        self.folderpath = [os.path.join(self.root,x) for x in os.listdir(self.root)]
        self.folderpath.sort()
        self.imgpath = []
        for i in self.folderpath:
            for j in os.listdir(i):
                self.imgpath.append(os.path.join(i,j))

        self.imgpath.sort()


    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = cv2.imread(self.imgpath[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert it to hsv
        hsv[...,2] = hsv[...,2]*0.05
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        img = np.array(img,dtype=np.float32)    # original image
        rgb = np.array(rgb,dtype=np.float32)    #low light rgb
        hsv = np.array(hsv,dtype=np.float32)    #low light hsv

        # imgf = np.fft.fft2(img)
        # imgshift = np.fft.fftshift(imgf)
        # ms = 20*np.log(np.abs(imgshift))



        if self.transforms_img is not None:
            img = self.transforms_img(img)
            rgb = self.transforms_img(rgb)
            hsv = self.transforms_img(hsv)

        # if self.transforms_freq is not None:
        #     ms = self.transforms_freq(ms)


        return (rgb,hsv,img)












        

