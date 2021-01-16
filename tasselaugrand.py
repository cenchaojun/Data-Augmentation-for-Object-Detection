#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 下午3:30
# @Author  : cenchaojun
# @File    : bbaug.py
# @Software: PyCharm
from bbaug import policies
import cv2
from PIL import Image, ImageDraw #version 6.1.0
import torchvision.transforms.functional as F
import PIL #version 1.2.0
import xml.etree.ElementTree as ET
import random
import torch
import numpy as np
from bbaug.visuals import visualise_policy
voc_labels = ('tassel',)
label_map = {k: v+1 for v, k in enumerate(voc_labels)}
#Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}
#Colormap for bounding box
CLASSES = 1
distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map  = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
def save(image, boxes, labels,image_name,police):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy=boxes[i], outline='#FCE38A',width=5)
    new_image.show()
    new_image.save('./geometric/{0}/{1}'.format(police,image_name),quality=100)

def parse_annot(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")
        label = object.find("name").text.lower().strip()
        if label not in label_map:
            print("{0} not in label map.".format(label))
            assert label in label_map

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}

# select policy v3 set
aug_policy = policies.policies_v3()
# print(aug_policy[2])

# instantiate the policy container with the selected policy set
policy_container = policies.PolicyContainer(aug_policy)
# augment = aug_policy[7]
# select a random policy from the policy set

random_policy = policy_container.select_random_policy()
# print(random_policy)

image = cv2.imread("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.jpg")
image_name = '10m_DJI_0075_5_0_1024_1024_2048.jpg'
# image = image.convert("RGB")
objects = parse_annot("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.xml")
bounding_boxes = torch.FloatTensor(objects['boxes'])
labels = torch.LongTensor(objects['labels'])
difficulties = torch.ByteTensor(objects['difficulties'])

# Apply the augmentation. Returns the augmented image and bounding boxes.
# Image is a numpy array of the image
# Bounding boxes is a list of list of bounding boxes in pixels (int).
# e.g. [[x_min, y_min, x_man, y_max], [x_min, y_min, x_max, y_max]]
# Labels are the class labels for the bounding boxes as an iterable of ints e.g. [1,0]
img_aug, bbs_aug = policy_container.apply_augmentation(random_policy, image, bounding_boxes, labels)
bbs_aug = np.delete(bbs_aug,0,axis=1)
save(img_aug, bbs_aug, labels,image_name=image_name,police='test')


# image_aug: numpy array of the augmented image
# bbs_aug: numpy array of augmneted bounding boxes in format: [[label, x_min, y_min, x_man, y_max],...]
# visualise_policy(
#     './data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.jpg',
#     './data',
#     bounding_boxes, # Bounding boxes is a list of list of bounding boxes in pixels (int): e.g. [[x_min, y_min, x_man, y_max], [x_min, y_min, x_man, y_max]]
#     labels, # Class labels for the bounding boxes as an iterable of ints eg. [0, 5]
#     policy, # the policy to visualise
#     name_to_augmentation, # (optional, default: augmentations.NAME_TO_AUGMENTATION) The dictionary mapping the augmentation name to the augmentation method
# )