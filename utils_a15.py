import numpy as np
import PIL.Image as Image
from tqdm import tqdm
from torch.utils import data
from PIL import Image
import zipfile
from zipfile import ZipFile
import glob

file_name='/content/gdrive/My Drive/Colab Notebooks/S14/DenseDepth/bgfg_images.zip'
file_name_mask='/content/gdrive/My Drive/Colab Notebooks/S14/DenseDepth/fgmask_images.zip'
file_name_depth='/content/gdrive/My Drive/Colab Notebooks/S14/DenseDepth/depthimages.zip'
file_name_bg='/content/gdrive/My Drive/Colab Notebooks/S14/DenseDepth/background/'

def make_dataset_bgfg(file_name,rndlist):
    images = []
    with ZipFile(file_name, 'r') as zip:
      image_files=zip.namelist()
      for i in rndlist:
        images.append(image_files[i])
    print("Required BGFG images loaded")
    return images

def make_dataset_mask(file_name_mask,rndlist):
    images = []
    with ZipFile(file_name_mask, 'r') as zip:
      image_files=zip.namelist()
      for i in rndlist: 
        images.append(image_files[i])
    print("Required MASK images loaded")
    return images

def make_dataset_depth(file_name_depth,rndlist):
    images = []
    with ZipFile(file_name_depth, 'r') as zip:
      image_files=zip.namelist()
      # print(len(image_files))
      for i in rndlist: 
        images.append(image_files[i])
    print("Required DEPTH images loaded")
    return images


from albumentations import  ( 
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose , Normalize ,ToFloat, Cutout
)

import cv2

import numpy as np

from albumentations.pytorch import  ToTensor 


# def downloading_data_transforms_albumentations(data_set):
class album_Compose_train():
    def __init__(self):
        self.albumentations_transform_train = Compose([
          # HorizontalFlip(),
          
          Cutout(num_holes=8),
          # CLAHE(),
          Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
          ),
          ToTensor()
        ])

    def __call__(self,img):
      img = np.array(img)
      img = self.albumentations_transform_train(image=img)
      return img['image']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torchsummary import summary


class Image_Dataset():
    def __init__(self, file_name,file_name_mask,file_name_depth,randlist,is_transform=False):
        self.file_name = file_name
        self.file_name_mask = file_name_mask
        self.file_name_depth = file_name_depth
        self.is_transform = is_transform
        self.randlist=randlist
        self.rgb_transform =album_Compose_train()
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.files_bgfg = make_dataset_bgfg(self.file_name,self.randlist)
        self.files_mask = make_dataset_mask(self.file_name_mask,self.randlist)
        self.files_depth = make_dataset_depth(self.file_name_depth,self.randlist)

    def __len__(self):
        return len(self.files_bgfg)
    
    
        

    def __getitem__(self, index):

        #put all the RGB images into one folder, and mask into another folder
        rgb_path = self.files_bgfg[index]
        bgp=rgb_path.split('/')[-1].split('_')[0]
        # print(bgp)
        mask_path=rgb_path.split('/')[-1].split('.jpg')[0] 
       
        with ZipFile(file_name, 'r') as zip1:
          with ZipFile(file_name_mask, 'r') as zip2:
            with ZipFile(file_name_depth, 'r') as zip3:
              imgg = Image.open(zip1.open(rgb_path)).convert('RGB')
              imgs = np.asarray(imgg)
              # img = transforms.ToPILImage()(img)
              # img=torch.from_numpy(img).long()
              # print(img.dtype)
              if self.is_transform:
                img = self.transform_img(imgs)
                # print(img.size(),"img reading")
              # print(glob.glob(file_name_bg+bgp+'.jpg')[0])
              bg_fg = Image.open(glob.glob(file_name_bg+bgp+'.jpg')[0]).convert('RGB')
              bgs = np.asarray(bg_fg)
              # bg=torch.from_numpy(bg).long()
              if self.is_transform:
                bg = self.transform_img(bgs)
                # print(bgp,"img reading")
              # print(bg.shape)
              bgimg= torch.cat([bg, img], axis=0)
              # print(bgimg.shape,'shape of bg,bg_fg')
              # print('foreground_mask/'+mask_path+'_M.jpg')
              masks = Image.open(zip2.open('foreground_mask/'+mask_path+'_M.jpg')).convert('L')
              maskk = np.asarray(masks)
              # mask=torch.from_numpy(mask).long()
              # print(mask.shape)
              if self.is_transform:
                msk = self.transform_mask(maskk)
                # print(msk.size(),"msk reading")
              deps = Image.open(zip3.open('depth/'+mask_path+'_D.jpg')).convert('L')
              depk = np.asarray(deps)
              # mask=torch.from_numpy(mask).long()
              # print(mask.shape)
              if self.is_transform:
                dep = self.transform_mask(depk)
          
          return bgimg,msk,dep

    def transform_img(self, imgs):
      img = self.rgb_transform(imgs)
    # msk = torch.FloatTensor(mask)
      return img
    def transform_mask(self,mskk):
      msk = self.mask_transform(mskk)
    # msk = torch.FloatTensor(mask)
      return msk



def Dataloader(traindata,bs):

  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  # For reproducibility
  SEED=1
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

      trainloader = torch.utils.data.DataLoader(traindata, batch_size=bs,
                                            shuffle=True, num_workers=4)
    
  print('Train data loaded.......')

  return trainloader



