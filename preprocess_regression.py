#%%
import os
from pathlib import Path
import pandas as pd
import shutil

def move_images(images, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    source_path = path_to_data/'training_data/regression'
    for image_num in images:
        filename = f"{image_num}.png"
        source_file_path = os.path.join(source_path, filename)
        destination_file_path = os.path.join(destination_path, filename)
        try:
            shutil.move(source_file_path, destination_file_path)
        except FileNotFoundError:
            pass
            
        
path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
training_norm = pd.read_csv(path_to_data/'training_norm.csv')

groups = training_norm.groupby('angle')
dfs = []
for group_name, group_df in groups:
    dfs.append(group_df)
#%%
for i, df in enumerate(dfs):
    angle = dfs[i]['angle'].iloc[0]
    images = df['image_id'].tolist()
    move_images(images, path_to_data/f"training_data/regression/{angle}")
#%%