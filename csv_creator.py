import pandas as pd
import os


data = []

for  folder_name in os.listdir("lego_photos"):
    folder_path = os.path.join('lego_photos', folder_name)
    piece, color, plant = folder_name.split("_")

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        data.append({
            'path': image_path,
            'pieceid': piece,
            'colorid':color,
            'plant': plant,
        })

df = pd.DataFrame(data)

df.to_csv("lego_pieces.csv")