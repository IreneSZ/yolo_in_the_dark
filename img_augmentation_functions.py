#!/usr/bin/env python
# coding: utf-8

# ### TO DO:
# ### tilt
# ### gamma correction
# ### show illumination histogram
# ### illumination change? illumination map?
# ### add noise

from __future__ import print_function, division, absolute_import

import math
import numbers
import sys
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

# random padding
# resize the raw image to fit the desired size
def scale_size(w, h, raw_h, raw_w):
    if raw_h < h and raw_w < w:
        scaled_h, scaled_w = raw_h, raw_w # if raw image is smaller, keep its original h and w
    else: # compress along the dim which is proportionally larger
        scale_ratio = min(h/raw_h, w/raw_w)
        scaled_h, scaled_w = math.floor(raw_h * scale_ratio), math.floor(raw_w * scale_ratio)
    return (scaled_h, scaled_w)

# randomly assign the padding area
# padding with black
def random_padding(h, w, scaled_img):
    scaled_h, scaled_w = scaled_img.shape[0], scaled_img.shape[1]
    h_padding = max(0, h - scaled_h)
    w_padding = max(0, w - scaled_w)
    scaled_h, scaled_w = scaled_img.shape[0], scaled_img.shape[1]
    top = random.randint(0, h_padding)
    left = random.randint(0, w_padding)
    canvas = np.zeros((h, w, 3)).astype(int)
    canvas[top:top+scaled_h, left:left+scaled_w,:] = scaled_img
    return canvas, top, left # the padded image and the offsets

# update the bounding boxes, reading from original xml file
# output into txt files
def get_cordinates(filename):
    doc = pq(filename=filename, parser='xml')
    raw_list = []
    raw_list.append(doc('filename')[0].text)
    size = [int(doc('width')[0].text), int(doc('height')[0].text)]
    raw_list.append(size) # [width, height]
    class_name = doc('folder')[0].text

    for obj in doc('object'):
        obj_name = pq(obj)('name')[0].text
        if obj_name == class_name:
            boxes = pq(obj)('bndbox')
            assert len(boxes) == 1
            xylist = [int(boxes('xmin')[0].text),
                     int(boxes('ymin')[0].text),
                     int(boxes('xmax')[0].text),
                     int(boxes('ymax')[0].text)]
            raw_list.append(xylist)
        else:
            pass
    return wrap(raw_list)

def wrap(raw_list):
    name = raw_list[0]
    size = Size(*raw_list[1])
    coor_lst = [Coor(*l) for l in raw_list[2:]]
    return Box(name, size, coor_lst)

def wrap_updated(raw_list, new_coor_list):
    name = raw_list[0]
    size = Size(*raw_list[1])
    coor_lst = [Coor(*l) for l in new_coor_list]
    return Box(name, size, coor_lst)

def update_coor_list(xml_path, scaled_ratio, top, left):
    coor_list = get_cordinates(xml_path)[2]
    new_coor_list = []
    for coor in coor_list:
        new_xmin = int(coor.xmin * scaled_ratio + left)
        new_xmax = int(coor.xmax * scaled_ratio + left)
        new_ymax = int(coor.ymax * scaled_ratio + top)
        new_ymin = int(coor.ymin * scaled_ratio + top)
        assert new_xmax <= w and new_ymax <= h
        new_xylist = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_coor_list.append(new_xylist) 
    return new_coor_list

def normalize_coor(box):
    bb_list = []
    for coor in box.coor_lst:
        class_name = box.name.split("_")[0]
        xcenter = (coor.xmin + coor.xmax) / 2 / box.size.w
        ycenter = (coor.ymin + coor.ymax) / 2 / box.size.h
        width = (coor.xmax - coor.xmin) / box.size.w
        height = (coor.ymax - coor.ymin) / box.size.h
        norm_coor = NormCoor(class_name, xcenter, ycenter, width, height)
        bb_list.append(norm_coor)
    return bb_list

def write_to_txt(bb_list, path):
    path = Path(path)
    file_name = bb_list[0].class_name
    with path.open("w") as fout:
        for coor in bb_list:
            fout.write(f"{file_name} {coor.xcenter} {coor.ycenter} {coor.width} {coor.height}\n")

# wrapper function
def data_prep(ID, w, h):
    """
    ID: the shared ID for the image and the bounding box
    w, h: the desired output image size
    """
    # resize and pad the image (if needed)
    img_path, xml_path = get_path(ID)[0], get_path(ID)[1]
    # output path
    new_img_path, new_txt_path = get_path(ID)[2], get_path(ID)[3]
    
    raw_img = cv2.imread(img_path)
    raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
    scaled_h, scaled_w = scale_size(w, h, raw_h, raw_w)
    scaled_img = cv2.resize(raw_img, (scaled_w, scaled_h))
    padded_img, top, left = random_padding(h, w, scaled_img)
    
    # save padded image
    cv2.imwrite(new_img_path, padded_img)
    
    # update the bounding box
    scaled_ratio = scaled_w / w
    raw_list = get_cordinates(xml_path)
    updated_list = update_coor_list(xml_path, scaled_ratio, top, left)
    box = wrap_updated(raw_list, updated_list)
    bb_list = normalize_coor(box)
    write_to_txt(bb_list, new_txt_path)

# make the image darker by adjusting gamma, bbox not affected
def adjust_gamma(ID, gamma):
    img_path = './image/' + ID + '.jpg'
    original = cv2.imread(img_path, 1)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(original, table)

# flip before padding
# output image has the same shape as the input image
# bounding boxes also flipped
def horizontal_flip(ID):
    img_path = './imge/' + ID + '.jpg'
    img = cv2.imread(img_path)
    img_output = np.flip(img, 1)
    # TO DO: flip the bbox
    
    return img_output               



