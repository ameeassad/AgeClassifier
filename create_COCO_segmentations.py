import json, yaml
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

from ultralytics import YOLO, utils
import random
import cv2
import os
from PIL import Image
import numpy as np

import matplotlib as plt
import pandas as pd

import ultralytics

import itertools
import json

from itertools import groupby
from skimage import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pycocotools.mask as mask_util

from shapely.geometry import Polygon

IMAGE_DIR = '/content/drive/MyDrive/artportalen_goeag'
YOLO_MODEL_PATH = 'content/yolov8x-seg.pt'

file_path_val = '/content/drive/MyDrive/artportalen_goleag_labels/val.csv'
file_path_train = '/content/drive/MyDrive/artportalen_goleag_labels/train.csv'
file_path_test = '/content/drive/MyDrive/artportalen_goleag_labels/test.csv'

class COCOArtportalen():
    def __init__(self, directory_path):
        self.IMAGE_DIR = directory_path
        self.info = {
                "year" : 2024,
                "version" : "1.0",
                "description" : "artportalen eagles",
                "contributor" : "Amee Assad",
                "url" : "",
                "date_created" : "2024"
        }
        self.images = []
        self.categories = []
        self.annotations = []

        self.label_mapping = {
                '1K': 1,
                '2K': 2,
                '3K': 3,
                '4K': 4,
                '5K_plus': 5,
                 }
        self.seg_count = 0

    def setup(self, file_path_train, file_path_val,file_path_test, choice):

        self.df_val = pd.read_csv(file_path_val)
        self.df_train = pd.read_csv(file_path_train)
        self.df_test = pd.read_csv(file_path_test)

        self.model = YOLO(YOLO_MODEL_PATH)

        self.set_df(choice)

    def set_df(self, choice):
        if choice == "example":
          self.df = self.df_train.head()
        elif choice == "test":
          self.df = self.df_test
        elif choice == "val":
          self.df = self.df_val
        elif choice == "train":
          self.df = self.df_train

        self.df['fileid'] = self.df['imageID'].astype('category').cat.codes
        self.df['categoryid'] = self.df['age_class'].map(self.label_mapping)

    def get_size(self, image_file):
        image = Image.open(os.path.join(self.IMAGE_DIR, str(image_file)))
        H, W = image.size
        return H, W

    def image(self, row):
        image = {}
        image["height"], image["width"] = self.get_size(row.imageID)
        image["id"] = row.fileid
        image["file_name"] = str(row.imageID)

        image["activity"] = str(row.activity)
        image["date_captured"] = str(row.date)
        image["photographer"] = str(row.Reporter)

        return image

    def category(self, row):
        category = {}
        category["supercategory"] = row.species
        category["id"] = row.categoryid
        category["name"] = row.age_class
        return category

    def get_segmentations(self, mask, image):
        mask = np.array(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = cocomask.merge(RLEs)
        # RLE = cocomask.encode(np.asfortranarray(mask))
        area = cocomask.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)
        return segmentation, [x, y, w, h], area

    def get_segs(self, polygon, image):
        segmentation = [polygon]

        poly = Polygon(polygon)
        area = poly.area
        min_x, min_y, max_x, max_y = poly.bounds
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        seg = polygon.flatten().tolist()
        # [x, y, w, h] = cv2.boundingRect(polygon)

        return seg, bbox, area

    def display_annotations(self,image,polygon=None):
      # Create a subplot
      fig, ax = plt.pyplot.subplots(1, 2, figsize=(10, 5))

      # Display the original image
      ax[0].imshow(image)
      ax[0].set_title('Original Image')
      ax[0].axis('off')

      if polygon is not None:
          # # Display the masked image
          # ax[1].imshow(mask_img)
          # ax[1].set_title('Segmentation Mask')
          # ax[1].axis('off')

          # Convert polygon points to integer
          polygon_points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)

          image_np_copy = np.array(image).copy()
          cv2.polylines(image_np_copy, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

          ax[1].imshow(image_np_copy)
          ax[1].set_title('Polygon')
          ax[1].axis('off')

          # Show the plot
          plt.pyplot.show()

    def get_annotations(self, row):
        temp_annotations = []
        image_file = str(row.imageID)

        image = Image.open(os.path.join(IMAGE_DIR, image_file))
        W, H = image.size
        results = self.model(image)
        for result in results:
            if result.masks is None:
                print("No mask found in result, printing result")
                self.display_annotations(image)
                continue
            for mask in result.masks:
                polygon = mask.xy[0]

            # if using masks instead of polygons and get_segmentations
            # for j, mask in enumerate(result.masks.data):
                # mask = mask.data[0]
                # mask_img = mask.numpy() * 255
                # mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)

                # FOR VISUALISATION - uncomment
                # self.display_annotations(image, polygon)

                segmentation, bbox, area = self.get_segs(polygon, image)
                # segmentation, bbox, area = self.get_segmentations(mask_img, image)

                iscrowd = 0 if len(results) <= 1 else 1
                annotation = {
                    'segmentation': [segmentation],
                    'bbox': bbox,
                    'area': int(area),
                    'image_id': row.fileid,
                    'category_id':row.categoryid,
                    'iscrowd': iscrowd,
                    'id': self.seg_count
                }
                temp_annotations.append(annotation)
                self.seg_count +=1
        return temp_annotations

    def fill_coco(self):
        i=0
        i_total = len(self.df)
        for row in self.df.itertuples():
            i+=1
            print("Row:", i, "/", i_total)
            self.annotations.extend(self.get_annotations(row))

        imagedf = self.df.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        for row in imagedf.itertuples():
            self.images.append(self.image(row))

        catdf = self.df.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
        for row in catdf.itertuples():
            self.categories.append(self.category(row))

    def create_coco_format_json(self, save_json_path = 'examplecoco.json'):
        data_coco = {}
        data_coco["info"] = self.info
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        json.dump(data_coco, open(save_json_path, 'w'), indent=4, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    Special json encoder for numpy types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def main():
    coco = COCOArtportalen(IMAGE_DIR)
    coco.setup(file_path_train, file_path_val, file_path_test, "val")
    coco.fill_coco()

    coco.create_coco_format_json("instances_val.json")

if __name__ == '__main__':
    main()