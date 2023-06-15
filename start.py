import os
import shutil
import sys
import itertools
import math
import logging
import json
import re
import random
import time
import concurrent.futures

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa
import pandas as pd

from samples.cell import cell

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log

config = cell.Config()

print(ROOT_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "samples//cell")
# Load dataset
dataset = cell.CellDataset()  # call the CellDataset class from the cell.py package

dataset.load_cell(DATASET_DIR, subset="train")

# Must call before using the dataset
dataset.prepare()  # prepares the dataset built-in function

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# image_ids = np.random.choice(dataset.image_ids, 10)
# print(image_ids)
#
# for image_id in image_ids:
#     print(image_id)
#     image = dataset.load_image(image_ids)
#     mask, class_ids = dataset.load_mask(image_ids)
#     # print(mask,mask.shape, class_ids)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)

# Поиск image_id по имени картинки
image_name = '90016c8865d3'
image_ids = dataset.image_ids
found_image_id = None
for image_id in image_ids:
    info = dataset.image_info[image_id]
    if info['id'] == image_name:
        found_image_id = image_id
        break

if found_image_id is not None:
    # Загрузка изображения и маски
    image = dataset.load_image(found_image_id)
    mask, class_ids = dataset.load_mask(found_image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)


# import glob
#
# folder_path = "E://CellInstanceSegmentation//SegmentationProj//Mask_RCNN//samples//cell//train"
# file_name = "90016c8865d3.png"
#
# # Получение списка файлов в папке
# files = glob.glob(folder_path + "/*.png")
#
# for i, file_path in enumerate(files):
#     if file_name in file_path:
#         file_number = i + 1
#         print(f"Файл {file_name} находится под номером {file_number} в папке {folder_path}.")
#         break
# else:
#     print(f"Файл {file_name} не найден в папке {folder_path}.")


# overlaying the original image & mask together
# Example of loading a specific image by its source ID
source_id = "90016c8865d3"

# Map source ID to Dataset image_id
# Notice the nucleus prefix: it's the name given to the dataset in NucleusDataset
image_id = dataset.image_from_source_map["cell.{}".format(source_id)]

# Load and display
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, config, image_id)
log("molded_image", image)
log("mask", mask)
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, show_bbox=False)

# ==================================================================================================

# Data path
TRAIN_PATH = '/samples/cell/train/'
TEST_PATH = '/samples/cell/test'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from release if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# ===========================================
#              DATASET LOADER
# ===========================================
source_dir = 'samples/cell/train'
target_dir = 'samples/cell/val_set'

file_names = os.listdir(source_dir)[:25]
for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), target_dir)

# Training dataset
dataset_train = cell.CellDataset()
dataset_train.load_cell(DATASET_DIR, subset="train")
dataset_train.prepare()

# Validation dataset
dataset_val = cell.CellDataset()
dataset_val.load_cell(DATASET_DIR, subset="val_set")
dataset_val.prepare()

# Create Model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Initialize coco weight
init_with = "coco"

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270)
    ]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

# Train the head
model.train(dataset_train, dataset_val, learning_rate=2 * config.LEARNING_RATE,
            epochs=5,
            augmentation=augmentation,
            layers='heads')

# Fine tune all layers
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10,
            epochs=25,
            augmentation=augmentation,
            layers='all')

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes3.h5")
model.keras_model.save_weights(model_path)

import glob
import skimage
import imageio

inference_config = cell.CellInferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model_infer = modellib.MaskRCNN(mode="inference", config=inference_config,
                                model_dir=MODEL_DIR)

model_path = '/content/logs/cell20220103T2332/mask_rcnn_cell_0002.h5'

# Load trained weights
model_infer.load_weights(model_path, by_name=True)

train = pd.read_csv('/content/cell/train.csv')

# Unique Image IDs
id_unique = train['id'].unique()


# Original Image File Path
def get_file_path(image_id):
    return f'/content/cell/train/{image_id}.png'


train['file_path'] = train['id'].apply(get_file_path)

# Unique Cell Names
CELL_NAMES = np.sort(train['cell_type'].unique())
print(f'CELL_NAMES: {CELL_NAMES}')

for file_path in glob.glob('/content/cell/train/*.png')[:25]:
    img = skimage.io.imread(file_path)
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=2)
    results = model_infer.detect([img], verbose=1)
    r = results[0]

    # Image Id
    image_id = file_path.split('/')[-1].split('.')[0]
    print(f'image_id: {image_id}')

    mask = cell.rle_decode(image_id, '/content/cell/train.csv')
    mask = np.sum(mask, axis=2)
    plt.figure(figsize=(16, 16))
    plt.imshow(mask)
    plt.show()

    visualize.display_instances(
        img,
        r['rois'],
        r['masks'],
        r['class_ids'],
        ['BG'] + CELL_NAMES.tolist(),
        r['scores'],
        figsize=(16, 16)
    )
