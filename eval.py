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


import torchmetrics
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def val(model, val_loader):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  valid_accuracy = Accuracy(average='macro', num_classes=10).to(device)
  valid_f1 = F1Score(average='macro', num_classes=10).to(device)
  valid_precision= Precision(average='macro', num_classes=10).to(device)
  valid_recall = Recall(average='macro', num_classes=10).to(device)
  model.eval() 
  with torch.no_grad(): 
        correct, total = 0, 0
        loop_val = tqdm(val_loader)
        for id_v, data_v in enumerate(loop_val):
            
          images, labels = data_v[0].float().to(device), data_v[1].to(device)
          images = transf(images)
	    
          test_output = model(images.float())
          pred_y = torch.max(test_output, 1)[1].data.squeeze()
          _, predicted = torch.max(test_output.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
            
          if labels.size(0) == 1:
	          metric_labels = labels[0]
          else:
            metric_labels = labels.squeeze()


          valid_accuracy.update(test_output, metric_labels)
          valid_f1.update(test_output, metric_labels)
          valid_precision.update(test_output, metric_labels)
          valid_recall.update(test_output, metric_labels)
	    
          loop_val.set_description(f"Validation]")     
          accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        
        total_valid_accuracy = valid_accuracy.compute()
        total_valid_f1 = valid_f1.compute()
        total_valid_precision = valid_precision.compute()
        total_valid_recall = valid_recall.compute()
        #print(f"Validation Acc,F1,Pr,Rec: {total_valid_accuracy, total_valid_f1, total_valid_precision, total_valid_recall}")
        print(f"\nAccuracy: {total_valid_accuracy.item():.2f}\n")
        print(f"F1Score {total_valid_f1.item():.2f}\n")
        print(f"Precision {total_valid_precision.item():.2f}\n")
        print(f"Recall {total_valid_recall.item():.2f}\n")
        valid_accuracy.reset()
        valid_f1.reset()
        valid_precision.reset()
        valid_recall.reset()
  torch.cuda.empty_cache()
  return 




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='image classification training')
  parser.add_argument('--val_path', type=str, default='imagewoof2-320/val', help='Val dataset path')
  args_opt = parser.parse_args()

  val_dataset = CustomDataset(args_opt.val_path)
  val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, drop_last=True)

  mean, std = get_mean_std(val_loader)
  transf = transforms.Compose([
     		transforms.Normalize(mean, std)])

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = models.convnext_base(pretrained=True)
  
  for param in model.parameters():
    param.requires_grad = False
  model.classifier[2] = nn.Linear(in_features=1024, out_features=10, bias=True)
  model.load_state_dict(torch.load('conv_next4.pth'))
  val(model, val_loader)

