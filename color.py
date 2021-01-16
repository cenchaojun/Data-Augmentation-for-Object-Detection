#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 上午10:11
# @Author  : cenchaojun
# @File    : augmentation.py
# @Software: PyCharm
from PIL import Image, ImageDraw,ImageOps #version 6.1.0
import PIL #version 1.2.0
import torch
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
from IPython.display import display
import cv2

voc_labels = ('tassel',)
label_map = {k: v+1 for v, k in enumerate(voc_labels)}
#Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}
#Colormap for bounding box
CLASSES = 1
distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map  = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


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
        draw.rectangle(xy=boxes[i], outline='#F38181',width=5)
    new_image.show()
    new_image.save('./color/{0}/{1}'.format(police,image_name),quality=100)
    # display(new_image)

image = cv2.imread("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.jpg")
image_name = '10m_DJI_0075_5_0_1024_1024_2048.jpg'
# image = image.convert("RGB")
objects = parse_annot("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.xml")
boxes = torch.FloatTensor(objects['boxes'])
labels = torch.LongTensor(objects['labels'])
difficulties = torch.ByteTensor(objects['difficulties'])

def Adjust_contrast(image):
    return F.adjust_contrast(image, 1.9)

def Adjust_brightness(image):
    return F.adjust_brightness(image, 1.9)

def Adjust_saturation(image):
    return F.adjust_saturation(image, 1.9)
def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image
    '''
    new_image = image
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms) - 1)]
    new_image = F.to_tensor(new_image)
    new_image = new_image[swap, :, :]
    new_image = F.to_pil_image(new_image)
    return new_image

def equalize(image):
    """
        均衡图像的直方图
    """

    img = ImageOps.equalize(image)
    return np.array(img)
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
def color(img, factor):
    """Equivalent of PIL Color."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    degenerate = np.tile(rgb2gray(img)[..., np.newaxis], [1, 1, 3])
    return blend(degenerate, img, factor)

def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.
    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.astype(np.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return np.clip(temp, 0.0, 255.0).astype(np.uint8)

# 调节对比度的，应该没有用到
new_image = Adjust_contrast(image)
save(new_image, boxes, labels,image_name=image_name,police='Adjust_contrast')

# 调节亮度的，也没有效果
new_image = Adjust_brightness(image)
save(new_image, boxes, labels,image_name=image_name,police='Adjust_brightness')

# 调节饱和段的
new_image = Adjust_saturation(image)
save(new_image, boxes, labels,image_name=image_name,police='Adjust_saturation')
# 增加光噪声
new_image = lighting_noise(image)
save(new_image, boxes, labels,image_name=image_name,police='lighting_noise')
# 白平衡 比较有用
new_image = equalize(image)
save(new_image, boxes, labels,image_name=image_name,police='equalize')
# # color (1,6) factor=6 这个是有用的
new_image = color(img=image,factor=6)
save(new_image, boxes, labels,image_name=image_name,police='color')