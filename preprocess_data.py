#%%
import os
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError
import multiprocessing as mp
import pandas as pd

path_to_data = Path(__file__).parent / f"./machine-learning-in-science-ii-2023"
training_norm = pd.read_csv(path_to_data/'training_norm.csv')

def image_generator(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            yield os.path.join(directory, filename)

def load_image(full_path):
    try:
        Image.open(full_path)
    except UnidentifiedImageError:
        delete_image(full_path)
        
def delete_image(full_path):
    if os.path.exists(full_path):
        # Delete the file
        remove_from_csv(full_path)
        os.remove(full_path)
        print(f"{full_path} deleted successfully.")
    else:
        # Raise an exception if the file doesn't exist
        raise FileNotFoundError(f"{full_path} not found.")
    
def remove_from_csv(full_path):
    filename = os.path.basename(full_path)
    image_num = int(os.path.splitext(filename)[0])
    image_index = training_norm.loc[training_norm['image_id'] == image_num].index
    try:
        training_norm.drop(image_index[0], inplace=True)
        print(f"Row for image '{image_num}.png' with index {image_index[0]} dropped from 'training_norm.csv'")
    except KeyError and IndexError:
        pass

def filter_corrupted_images(directory):
    pool = mp.Pool()
    full_paths = image_generator(directory)
    pool.map(load_image, full_paths)
    pool.close()
    pool.join()
    
def filter_non_integer_speeds(directory):
    indexes = training_norm.loc[training_norm['speed']
                                .apply(lambda x: not x.is_integer())].index
    non_integer_indexes = []
    non_integer_image_ids = []
    if len(indexes) > 1:
        for index in indexes:
            non_integer_indexes.append(index[0])
            non_integer_image_ids.append(int(training_norm.iloc[index[0]].image_id))
    else:
        non_integer_indexes.append(indexes[0])
        non_integer_image_ids.append(int(training_norm.iloc[indexes[0]].image_id))
    for i, image_id in enumerate(non_integer_image_ids):
        delete_image(directory/f"{image_id}.png")

def filter_images(directory):
    filter_corrupted_images(directory)
    filter_non_integer_speeds(directory)
    
filter_images(path_to_data/'training_data/training_data')
#%%
def create_category_csvs(group):
    speed = group['speed'].iloc[0]
    file = f"training_norm_{speed}.csv"
    group.to_csv(os.path.join(path_to_data, file), index=False)

training_norm.groupby(['speed']).apply(create_category_csvs)
speed_zero = pd.read_csv(path_to_data/'training_norm_0.0.csv')
speed_one = pd.read_csv(path_to_data/'training_norm_1.0.csv')
#%%
import shutil

source_path = path_to_data/'training_data/training_data'
destination_path = path_to_zeros

for filename in os.listdir(source_path):
    if filename.endswith('.png'):
        # construct the paths for the source and destination files
        source_file_path = os.path.join(source_path, filename)
        destination_file_path = os.path.join(destination_path, filename)
        
        # move the file to the destination folder
        shutil.move(source_file_path, destination_file_path)
#%%