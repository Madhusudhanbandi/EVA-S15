
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torchsummary import summary
from pytorch_msssim import ssim
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


train_losses = []
train_acces = []
train_accesd = []

def train(model, device, train_loader, optimizer, epoch,EPOCHS,criterion,cos):

  pred_images=torch.cuda.FloatTensor()
  pred_images_dep=torch.cuda.FloatTensor()
  model.train()
  # pbar = tqdm(train_loader)
  loss = 0
  train_loss=0
  train_acc=0
  train_accd=0
  processed = 0
  # for batch_idx, (data, target) in enumerate(pbar):
    # get samples
  for data, target,depth in train_loader:
    data, target,depth = data.to(device), target.to(device),depth.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred_mask,y_pred_dep = model(data)
    # print(y_pred)
    pred_dep = y_pred_dep.squeeze()
    pred_mask = y_pred_mask.squeeze()
    # print(pred_mask.shape)
    pred_maskl = pred_mask.view(-1)
    real_mask = target.view(-1)

    if epoch==EPOCHS-1:
      
      pred_images=torch.cat((pred_images,pred_mask),dim=0)
      pred_images_dep=torch.cat((pred_images_dep,pred_dep),dim=0)
    
    else:
      pred_images=pred_images
      pred_images_dep=pred_images_dep
    

    pred_depl = pred_dep.view(-1)
    real_depth = depth.view(-1)
    # Calculate loss
    loss1 = criterion(pred_maskl, real_mask)
    # print("Loss",loss)
    loss2 = criterion(pred_depl, real_depth)
    
    loss=loss1+2*loss2

    # loss=loss1

    # acc= cos(pred_maskl, real_mask)
    # acc_sim= ssim(target, target)
    acc_sim= ssim(y_pred_mask, target)
    acc_simd= ssim(y_pred_dep, depth)
    # print("Acc",acc_sim)
    # print("Acc",acc_simd)
    # Backpropagation
    loss.backward()
    optimizer.step()
    # print(loss.item())
    train_loss +=loss.item()
    train_acc +=acc_sim.item()
    train_accd +=acc_simd.item()
  # print(train_loss)
  train_loss /= len(train_loader.dataset)
  train_acc /= len(train_loader.dataset)
  train_accd /= len(train_loader.dataset)
  train_losses.append(train_loss)
  train_acces.append(train_acc)
  train_accesd.append(train_accd)
  print('\nTrain Average BCE loss: {:.4f}'.format(train_loss))
  print('\nTrain Average SSIM Accuracy for mask images: {:.4f}'.format(train_acc))
  print('\nTrain Average SSIM Accuracy for depth images: {:.4f}'.format(train_accd))

  return pred_images,pred_images_dep,train_losses,train_acces,train_accesd
    
    