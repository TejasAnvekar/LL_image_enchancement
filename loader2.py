import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import random
random.seed(1)



def sp(img,prob):
    out=np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                rdn =random.random()
                if rdn <prob:
                    out[i,j,k]=0
                if rdn > prob:
                    out[i,j,k]=255

    return out

class lowlight(Dataset):
    def __init__(self,A,B,transforms=None):
        super(lowlight,self).__init__()
        self.A = A
        self.B = B
        self.transforms = transforms
        self.folderA = os.listdir(self.A)
        self.folderB = os.listdir(self.B)

        self.folderA.sort()
        self.folderB.sort()

        self.Aimgpath=[]
        self.Bimgpath=[]

        for i in range(len(self.folderA)):
            p=os.path.join(self.A,self.folderA[i])
            for j in os.listdir(p):
                self.Aimgpath.append(os.path.join(p,j))

        for i in range(len(self.folderB)):
            p=os.path.join(self.B,self.folderB[i])
            for j in os.listdir(p):
                self.Bimgpath.append(os.path.join(p,j))

        self.Aimgpath.sort()
        self.Bimgpath.sort()


    def __len__(self):
        return len(self.Aimgpath)

    def __getitem__(self,index):
        A = cv2.imread(self.Aimgpath[index])
        B = cv2.imread(self.Bimgpath[index])

        # A = cv2.cvtColor(A,cv2.COLOR_BGR2RGB)
        # B = cv2.cvtColor(B,cv2.COLOR_BGR2RGB)

       # num=np.random.uniform(low=0.5,high=1,size=1)

       # A = sp(A,num)
        A1 = cv2.cvtColor(A,cv2.COLOR_BGR2HSV)
        A = np.array(A)
        A1 = np.array(A1)
        B = np.array(B)


        if self.transforms is not None:
            A = self.transforms(A)
            A1 = self.transforms(A1)
            B = self.transforms(B)

        
        return (A,A1,B)



