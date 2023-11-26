import os
import pandas as pd
from PIL import Image
from .description import *
from .segmentations import region_growing
from tqdm import tqdm

def crop_with_pil(image, x, y, crop_size):
    lcrop = int(crop_size/2)
    rcrop = int(crop_size - lcrop)
    width, height = image.size

    x_start = max(0, x - lcrop)
    y_start = max(0, y - lcrop)

    x_end = min(width, x + rcrop)
    y_end = min(height, y + rcrop)

    cropped_image = image.crop((x_start, y_start, x_end, y_end))
    return cropped_image

def crop_dataset(df:pd.DataFrame, dir_path:str, crop_size=100, out_dir='./cropped_dataset/'):
    for index, row in df.iterrows():
        x = row.nucleus_x
        y = row.nucleus_y
        id = row.cell_id
        label = row.bethesda_system
        img_path = os.path.join(dir_path, row.image_filename)
        if os.path.isdir(os.path.join(out_dir, label)) == False:
            os.mkdir(os.path.join(out_dir, label))
        
        cropped_img_path = os.path.join(out_dir, label, str(id) +'.jpg')

        image = Image.open(img_path)
        cropped_img = crop_with_pil(image, x, y, crop_size)
        cropped_img.save(cropped_img_path)

def generate_cropped_data(from_dir='./cropped_dataset/'):
    df = pd.DataFrame(columns=['image', 'area', 'compactness', 'eccentricity', 'permieter', 'label'])
    error_log = []
    for dir in (os.listdir(from_dir)):
        for file in tqdm(os.listdir(from_dir + "/" + dir)):
            try:
                path = (from_dir + dir + "/" + file)

                img = cv2.imread(path)
                test = region_growing(img)
                test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                (x, y) = test.shape

                main_obj = test[x//2, y//2]

                test[test != main_obj] = 0
                test[test == main_obj] = 255

                df.loc[len(df.index)] = ((path, 
                    calculate_area(test),
                    calculate_compactness(test),
                    calculate_eccentricity(test),
                    calculate_perimeter(test), 
                dir))
            except:
                print(f'error: {path}')
                error_log.append(path)
    df.to_csv('./main_data.csv')
    return error_log, df