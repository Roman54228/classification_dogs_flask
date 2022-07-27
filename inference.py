from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import time
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from custom_dataset import CustomDataset, get_mean_std
import argparse
import ast
from PIL import Image
import requests
from io import BytesIO


import torchmetrics
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



class_names = {0: 'Shih-Tzu',
 1: 'Rhodesian_ridgeback',
 2: 'beagle',
 3: 'English_foxhound',
 4: 'Border_terrier',
 5: 'Australian_terrier',
 6: 'golden_retriever',
 7: 'Old_English_sheepdog',
 8: 'Samoyed',
 9: 'dingo'}

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='image classification inference')
  parser.add_argument('--image_url', type=str, default='https://www.purina.com.au/-/media/project/purina/main/breeds/dog/dog_samoyed_desktop.jpg', help='url of dog image')
  args_opt = parser.parse_args()

  response = requests.get(args_opt.image_url)
  img = Image.open(BytesIO(response.content)) 


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = models.convnext_base(pretrained=True)
  
  for param in model.parameters():
    param.requires_grad = False
  model.classifier[2] = nn.Linear(in_features=1024, out_features=10, bias=True)
  model.load_state_dict(torch.load('/content/classification_dogs_flask/conv_next4.pth'))
  trans = transforms.ToTensor()
  img = trans(img).to(device)
  
  model.eval()
  model.to(device)
  with torch.no_grad():
      outputs = model(img.unsqueeze(0).float())
  _, predicted = torch.max(outputs.data,1)
  
  class_name = class_names[predicted.item()]
  prob = float(outputs[0][predicted.item()])

  print(f"\n\nCLASS: {class_name}, PROB: {prob} ")

