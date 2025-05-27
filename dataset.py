import pandas as pd
from torch.utils.data import Dataset
from resizer import resizer
import cv2
import torch


class MyDataset(Dataset):
    def __init__(self, labels_csv, transform=resizer):
        self.labels = pd.read_csv(labels_csv)
        self.transform = transform

        #create a dictionary linking id to an int for piece and color (id -> int)
        self.pieceid_dict = {v: i for i, v in enumerate(self.labels['pieceid'].unique())}
        self.colorid_dict = {v: i for i, v in enumerate(self.labels['colorid'].unique())}

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        row = self.labels.iloc[index] #get the row we want to load the data from
        img_path = row['path']

        
        image = cv2.imread(img_path)

        #use the id -> int dicts to get the piece and color num, plants is a string with value 0/1 so convert it to int
        piece_label = self.pieceid_dict[row['pieceid']]
        color_label = self.colorid_dict[row["colorid"]]
        plant_label = int(row["plant"])

        image = self.transform(image)

        # retrun the image and a dict linking pieceid, colorid, plant to a tensor made from the label (if its not a tensor it will raise an error),
        #the dict isnt needed (could be a list), but simplifies calling for the labels in main
        return image, {
            "pieceid": torch.tensor(piece_label, dtype=torch.int64), 
            "colorid": torch.tensor(color_label, dtype=torch.int64),
            "plant": torch.tensor(plant_label, dtype=torch.float32),
        }