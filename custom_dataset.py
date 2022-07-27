import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from tqdm import tqdm

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data = data.float()
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

class CustomDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "/*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.JPEG"):
                self.data.append([img_path, class_name])
        self.class_map = {'n02086240': 0,
 'n02087394': 1,
 'n02088364': 2,
 'n02089973': 3,
 'n02093754': 4,
 'n02096294': 5,
 'n02099601': 6,
 'n02105641': 7,
 'n02111889': 8,
 'n02115641': 9}
        self.img_dim = (416, 416)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id
