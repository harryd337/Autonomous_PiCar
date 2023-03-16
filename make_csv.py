#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
#%%
# LOCAL - TRAIN
path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
training_norm = pd.read_csv(path_to_data/'training_norm.csv')

def add_substring(image_id):
    full_path = path_to_data/'training_data/combined'/f"{str(image_id)}.png"
    if not os.path.exists(full_path):
        remove_from_csv(full_path)
    return full_path

def remove_from_csv(full_path):
    filename = os.path.basename(full_path)
    image_num = int(os.path.splitext(filename)[0])
    image_index = training_norm.loc[training_norm['image_id'] == image_num].index
    try:
        training_norm.drop(image_index[0], inplace=True)
        print(f"Row for image '{image_num}.png' with index {image_index[0]} dropped from 'training_norm.csv'")
    except KeyError and IndexError:
        pass

# apply the function to the values in the 'column2' column of the dataframe
training_norm['image_id'] = training_norm['image_id'].apply(lambda x: add_substring(x))

training_norm.rename(columns={'image_id': 'image_path'}, inplace=True)
training_norm.to_csv(path_to_data/'training_norm_paths.csv', index=False)

training_norm_paths = pd.read_csv(path_to_data/'training_norm_paths.csv')
#%%
# LOCAL - TEST
path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"

# apply the function to the values in the 'column2' column of the dataframe
image_ids = np.arange(1, 1021)
test_image_paths = pd.DataFrame({'image_path': []})
for image_id in image_ids:
    full_path = path_to_data/'test_data/test_data'/f"{str(image_id)}.png"
    test_image_paths.loc[len(test_image_paths)] = full_path

test_image_paths.to_csv(path_to_data/'test_image_paths.csv', index=False)
#%%
# GOOGLE DRIVE - TRAIN
path_to_googledrive_data = Path('/content/drive/MyDrive/machine-learning-in-science-ii-2023')
training_norm = pd.read_csv(path_to_data/'training_norm.csv')

def add_substring(image_id):
    return path_to_googledrive_data/'training_data/combined'/f"{str(image_id)}.png"

# apply the function to the values in the 'column2' column of the dataframe
training_norm['image_id'] = training_norm['image_id'].apply(lambda x: add_substring(x))

training_norm.rename(columns={'image_id': 'image_path'}, inplace=True)
training_norm.to_csv(path_to_data/'training_norm_paths_googledrive.csv', index=False)

training_norm_paths = pd.read_csv(path_to_data/'training_norm_paths_googledrive.csv')
# %%
# GOOGLE DRIVE - TEST
path_to_googledrive_data = Path('/content/drive/MyDrive/machine-learning-in-science-ii-2023')
path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"

# apply the function to the values in the 'column2' column of the dataframe
image_ids = np.arange(1, 1021)
test_image_paths = pd.DataFrame({'image_path': []})
for image_id in image_ids:
    full_path = path_to_googledrive_data/'test_data/test_data'/f"{str(image_id)}.png"
    test_image_paths.loc[len(test_image_paths)] = full_path

test_image_paths.to_csv(path_to_data/'test_image_paths_googledrive.csv', index=False)
# %%
