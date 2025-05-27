import pandas as pd
from torch.utils.data import Dataset
from resizer import resizer
import cv2

class MyDataset(Dataset):
    def __init__(self, labels_csv, transform=resizer):
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.labels.iloc[index]['path']
        image = cv2.imread(img_path)

        label = {
            "pieceid": self.labels.iloc[index]['pieceid'],
            "color": self.labels.iloc[index]['colorid'],
            "plant": int(self.labels.iloc[index]['plant'])
        }

        if self.transform:
            image = self.transform(image)

        return image, label