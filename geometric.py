#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 上午10:11
# @Author  : cenchaojun
# @File    : augmentation.py
# @Software: PyCharm
from PIL import Image, ImageDraw #version 6.1.0
import PIL #version 1.2.0
import torch
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2
from IPython.display import display

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
        draw.rectangle(xy=boxes[i], outline='#FCE38A',width=5)
    new_image.show()
    new_image.save('./geometric/{0}/{1}'.format(police,image_name),quality=100)
    # display(new_image)

# 水平反转
def flip(image, boxes):
    '''
        Flip image horizontally.
        image: a PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
    '''
    new_image = F.hflip(image)

    # flip boxes
    new_boxes = boxes.clone()
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes
# 旋转
def rotate(image, boxes, angle):
    '''
        Rotate image and bounding box
        image: A Pil image (w, h)
        boxes: A tensors of dimensions (#objects, 4)

        Out: rotated image (w, h), rotated boxes
    '''
    new_image = image.copy()
    new_boxes = boxes.clone()

    # Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w / 2
    cy = h / 2
    new_image = new_image.rotate(angle, expand=True)
    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    # Get affine matrix
    AffineMatrix = torch.tensor([[alpha, beta, (1 - alpha) * cx - beta * cy],
                                 [-beta, alpha, beta * cx + (1 - alpha) * cy]])

    # Rotation boxes
    box_width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
    box_height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

    # Get corners for boxes
    x1 = boxes[:, 0].reshape(-1, 1)
    y1 = boxes[:, 1].reshape(-1, 1)

    x2 = x1 + box_width
    y2 = y1

    x3 = x1
    y3 = y1 + box_height

    x4 = boxes[:, 2].reshape(-1, 1)
    y4 = boxes[:, 3].reshape(-1, 1)

    corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    corners.reshape(8, 8)  # Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1, 2)  # Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim=1)  # (Tensors of dimension (4* #objects, 3))

    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy

    # Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t()).t()
    rotate_corners = rotate_corners.reshape(-1, 8)

    x_corners = rotate_corners[:, [0, 2, 4, 6]]
    y_corners = rotate_corners[:, [1, 3, 5, 7]]

    # Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim=1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim=1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim=1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim=1)
    y_max = y_max.reshape(-1, 1)

    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim=1)

    scale_x = new_image.width / w
    scale_y = new_image.height / h

    # Resize new image to (w, h)
    new_image = new_image.resize((500, 333))

    # Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    return new_image, new_boxes

def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)

def find_IoU(boxes1, boxes2):
    '''
        Find IoU between every boxes set of boxes
        boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
        boxes2: a tensor of dimensions (n2, 4)

        Out: IoU each of boxes1 with respect to each of boxes2, a tensor of
             dimensions (n1, n2)

        Formula:
        (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    '''
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter)  # (n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  # (n1, n2)
    union = (area_boxes1 + area_boxes2 - inter)
    return inter / union

# 随机裁剪
def random_crop(image, boxes, labels, difficulties):
    '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        difficulties: difficulties of detect object, a tensor of dimensions (#objects)

        Out: cropped image , new boxes, new labels, new difficulties
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)

    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        if mode is None:
            return image, boxes, labels, difficulties

        new_image = image
        new_boxes = boxes
        new_difficulties = difficulties
        new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            new_h = random.uniform(0.3 * original_h, original_h)
            new_w = random.uniform(0.3 * original_w, original_w)

            # Aspect ratio constraint b/t .5 & 2
            if new_h / new_w < 0.5 or new_h / new_w > 2:
                continue

            # Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), boxes)  # (1, #objects)
            overlap = overlap.squeeze(0)
            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.max().item() < mode:
                continue

            # Crop
            new_image = image[:, int(top):int(bottom), int(left):int(right)]  # (3, new_h, new_w)

            # Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # Find bounding box has been had center in crop
            center_in_crop = (center_bb[:, 0] > left) * (center_bb[:, 0] < right
                                                         ) * (center_bb[:, 1] > top) * (
                                         center_bb[:, 1] < bottom)  # ( #objects)

            if not center_in_crop.any():
                continue

            # take matching bounding box
            new_boxes = boxes[center_in_crop, :]

            # take matching labels
            new_labels = labels[center_in_crop]

            # take matching difficulities
            new_difficulties = difficulties[center_in_crop]

            # Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])

            # adjust to crop
            new_boxes[:, :2] -= crop[:2]

            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])

            # adjust to crop
            new_boxes[:, 2:] -= crop[:2]

            return F.to_pil_image(new_image), new_boxes, new_labels, new_difficulties
# 仅仅让bbbox旋转
def rotate_only_bboxes(image, boxes, angle):
    new_image = image.copy()
    new_image = F.to_tensor(new_image)
    for i in range(boxes.shape[0]):
        x_min, y_min, x_max, y_max = map(int, boxes[i, :].tolist())
        bbox = new_image[:, y_min:y_max + 1, x_min:x_max + 1]
        bbox = F.to_pil_image(bbox)
        bbox = bbox.rotate(angle)

        new_image[:, y_min:y_max + 1, x_min:x_max + 1] = F.to_tensor(bbox)
    return F.to_pil_image(new_image)

def translate_x(img, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)


    img = img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=replace)
    # img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, 0), fillcolor=replace)
    return np.array(img)
def translate_y(img, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(np.uint8(img))

    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=replace)
    return np.array(img)
def _clip_bbox(min_y, min_x, max_y, max_x):
    """Clip bounding box coordinates between 0 and 1.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
    Returns:
      Clipped coordinate values between 0 and 1.
    """
    min_y = np.clip(min_y, 0.0, 1.0)
    min_x = np.clip(min_x, 0.0, 1.0)
    max_y = np.clip(max_y, 0.0, 1.0)
    max_x = np.clip(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x
def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    """Adjusts bbox coordinates to make sure the area is > 0.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
      delta: Float, this is used to create a gap of size 2 * delta between
        bbox min/max coordinates that are the same on the boundary.
        This prevents the bbox from having an area of zero.
    Returns:
      Tuple of new bbox coordinates between 0 and 1 that will now have a
      guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        if min_coord == 0.0 and max_coord == 0.0:
            max_coord = delta
        elif min_coord == 1.0 and max_coord == 1.0:
            min_coord = 1 - delta
        else:
            max_coord = max(max_coord + delta, 1.0)
            min_coord = min(min_coord - delta, 0)
        return min_coord, max_coord

    if height == 0:
        min_y, max_y = _adjust_bbox_boundaries(min_y, max_y)

    if width == 0:
        min_x, max_x = _adjust_bbox_boundaries(min_x, max_x)

    return min_y, min_x, max_y, max_x
def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    """Shifts the bbox coordinates by pixels.
    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: Int, height of the image.
      image_width: Int, width of the image.
      pixels: An int. How many pixels to shift the bbox.
      shift_horizontal: Boolean. If true then shift in X dimension else shift in
        Y dimension.
    Returns:
      A tensor of the same shape as bbox, but now with the shifted coordinates.
    """
    pixels = pixels

    # Convert bbox to integer pixel locations.

    min_x = int(image_width * bbox[0])
    min_y = int(image_height * bbox[1])
    max_x = int(image_width * bbox[2])
    max_y = int(image_height * bbox[3])
    if shift_horizontal:
        min_x = min_x - pixels
        max_x = max_x - pixels
    else:
        min_y = min_y - pixels
        max_y = max_y - pixels


    # if shift_horizontal:
    #     min_x = np.maximum(0, min_x - pixels)
    #     max_x = np.minimum(image_width, max_x - pixels)
    # else:
    #     min_y = np.maximum(0, min_y - pixels)
    #     max_y = np.minimum(image_height, max_y - pixels)

    # Convert bbox back to floats.
    min_y = float(min_y) / float(image_height)
    min_x = float(min_x) / float(image_width)
    max_y = float(max_y) / float(image_height)
    max_x = float(max_x) / float(image_width)

    # if max_x < 0. or min_x > 1.0 or max_y < 0. or min_y > 1.0:
    #     return None

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    if min_x > max_x or min_y > max_y:
        print(min_x, min_y, max_x, max_y)
    return np.stack([min_x, min_y, max_x, max_y])
def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.
        Args:
          image: 3D uint8 Tensor.
          bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
            has 4 elements (min_x, min_y, max_x, max_y) of type float with values
            between [0, 1].
          pixels: An int. How many pixels to shift the image and bboxes
          replace: A one or three value 1D tensor to fill empty pixels.
          shift_horizontal: Boolean. If true then shift in X dimension else shift in
            Y dimension.
        Returns:
          A tuple containing a 3D uint8 Tensor that will be the result of translating
          image by pixels. The second element of the tuple is bboxes, where now
          the coordinates will be shifted to reflect the shifted image.
        """
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)
    print(boxes)
    # print(boxes[:,0:2])
    offset = torch.zeros(14,4)
    offset[:,0] = 100
    offset[:,2] = 100
    print(offset)
    new_box = boxes - offset
    print(new_box)
    # # Convert bbox coordinates to pixel values.
    # image_height = image.shape[0]
    # image_width = image.shape[1]
    # # pylint:disable=g-long-lambda
    # wrapped_shift_bbox = lambda bbox: _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)
    # # 格式正确
    # # pylint:enable=g-long-lambda
    # bboxes = np.array([box for box in list(map(wrapped_shift_bbox, bboxes)) if box is not None])
    return image, new_box


image = cv2.imread("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.jpg")
image_name = '10m_DJI_0075_5_0_1024_1024_2048.jpg'
# image = image.convert("RGB")
objects = parse_annot("./data/cloudaug/10m_DJI_0075_5_0_1024_1024_2048.xml")
boxes = torch.FloatTensor(objects['boxes'])
labels = torch.LongTensor(objects['labels'])
difficulties = torch.ByteTensor(objects['difficulties'])
new_image, new_boxes = translate_bbox(image, boxes,pixels=100,replace=None,shift_horizontal=True)
# new_new_boxes = torch.from_numpy(new_boxes) * boxes
#  返回的好像是偏移的比例，需要，改一下
save(new_image, new_boxes, labels,image_name=image_name,police='translate_bbox')

# # 水平反转
# new_image, new_boxes = flip(image, boxes)
# save(new_image, new_boxes, labels,image_name=image_name,police='flip')
# # 旋转
#
# new_image, new_boxes = rotate(image, boxes, 10)
# save(new_image, new_boxes, labels, image_name=image_name,police='rotate')

# ('TranslateX_BBox', 0.6, 4)
new_image, new_boxes = translate_bbox(image, boxes,pixels=100,replace=None,shift_horizontal=True)
save(new_image, new_boxes, labels,image_name=image_name,police='translate_bbox')

# ('TranslateY_Only_BBoxes', 0.2, 2)


# ('Cutout', 0.8, 8)
# ('ShearY_BBox', 1.0, 2)
# ('TranslateY_Only_BBoxes', 0.6, 6)

