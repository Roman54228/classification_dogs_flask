#llll
#bebe
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






def train(model, train_loader, val_loaderm, num_epochs=15):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.05)

  train_accuracy = Accuracy(average='macro', num_classes=10).to(device)
  valid_accuracy = Accuracy(average='macro', num_classes=10).to(device)
  valid_f1 = F1Score(average='macro', num_classes=10).to(device)
  valid_precision= Precision(average='macro', num_classes=10).to(device)
  valid_recall = Recall(average='macro', num_classes=10).to(device)

  transf = transforms.Compose([
                 transforms.RandomHorizontalFlip(p=0.6),
                 transforms.Normalize(mean, std)])
 
  model.to(device)
  model.train()
  for epoch in range(num_epochs):  
    running_loss = 0.0
    loop = tqdm(train_loader)
    for i, data in enumerate(loop):
        start_time = time.time()
        inputs, labels = data[0].float().to(device), data[1].to(device)
        inputs = transf(inputs)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward() 
        train_accuracy(outputs, labels.squeeze())

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
        
    PATH = './conv_next' + str(epoch) + '.pth'
    torch.save(model.state_dict(), PATH)
    
    total_train_accuracy = train_accuracy.compute() 
    print(f"Training acc for epoch {epoch}: {total_train_accuracy}")
    train_accuracy.reset()
    
     
    print(f"Finished epoch {epoch}\n")
    
    #start eval
    #----------------------------------------------------------------------------------------------------------------------------
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
	    
          loop_val.set_description(f"Val_Epoch [{epoch}/{num_epochs}]")     
          accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    

    total_valid_accuracy = valid_accuracy.compute()
    total_valid_f1 = valid_f1.compute()
    total_valid_precision = valid_precision.compute()
    total_valid_recall = valid_recall.compute()

    print(f"Validation Acc,F1,Pr,Rec for epoch {epoch}: {total_valid_accuracy, total_valid_f1, total_valid_precision, total_valid_recall}")
    valid_accuracy.reset()
    valid_f1.reset()
    valid_precision.reset()
    valid_recall.reset()
  torch.cuda.empty_cache()
  return




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='image classification training')
	parser.add_argument('--train_path', type=str, default='imagewoof2-320/train', help='Train dataset path')
	parser.add_argument('--val_path', type=str, default='imagewoof2-320/val', help='Val dataset path')
	parser.add_argument('--epoch_size', type=int, default=15, help='Number of epochs')
	args_opt = parser.parse_args()


	train_dataset = CustomDataset(args_opt.train_path)
	val_dataset = CustomDataset(args_opt.val_path)

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, drop_last=True)
	
	
	mean, std = get_mean_std(train_loader)
	transf = transforms.Compose([
    		transforms.RandomHorizontalFlip(p=0.6),
     		transforms.Normalize(mean, std)])
	

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = models.convnext_base(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False
	model.classifier[2] = nn.Linear(in_features=1024, out_features=10, bias=True)
	
	#start training
	#--------------------------------------------------------------------------------------------------------------------------------
	train(model, train_loader, val_loader, args_opt.epoch_size)
	
