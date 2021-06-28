import random
import cv2
import numpy as np
import json
from scipy.ndimage.interpolation import geometric_transform, map_coordinates
from scipy.ndimage.filters import gaussian_filter

from params import args
import coco_helper
import os
import math

import coco_helper as hp


input_width = args.input_width
input_height = args.input_height

input_rate = input_height / input_width

def random_geometry(image, points, bbox):
    bx, by, bw, bh = bbox
    base_corners = np.array(
        [
            [bx, by],
            [bx + bw, by],
            [bx + bw, by + bh],
            [bx, by + bh],
        ], dtype=np.float32)

    br = bh / bw
    #max scale
    if br > input_rate:
        scale0 = input_height / bh
    else:
        scale0 = input_width / bw
    # 60 % to 100
    scale = scale0 * 0.6 + scale0 * 0.4 * random.random()
    
    tw = bw * scale
    th = bh * scale
    x0 = (input_width - tw) * random.random()
    y0 = (input_height - th) * random.random()
    x1 = x0 + tw
    y1 = y0 + th


    #--코너점
    corners = np.array(
        [
            [x0, y0, 1],
            [x1, y0, 1],
            [x1, y1, 1],
            [x0, y1, 1]
        ], dtype=np.float32)

    #-- 회전 + (스케일)
    aug_angle = random.uniform(-20, 20)
    #ori_width is center of nu_image
    rot_matrix = cv2.getRotationMatrix2D(((x0 + x1) // 2, (y0 + y1) // 2), aug_angle, 1.0).astype(np.float32)
    corners = corners @ rot_matrix.T

    #-- 랜덤 추가
    corners += np.random.randn(4, 2) * (input_width / 60)

    #-- 호모그래피
    h = cv2.getAffineTransform(base_corners[:3], corners[:3]).astype(np.float32)

    dst_image = cv2.warpAffine(image, h, (input_width, input_height))
    s = np.concatenate([points, np.ones((17, 1), dtype=np.float32)], axis=1)
    points = (s @ h.T)[:, :2]

    return dst_image, points

def random_salt_and_papper(image):
    #salt and papper
    width = image.shape[1]
    height = image.shape[0]
    
    salts = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0]]

    for salt in salts:
        if random.random() < 0.1:
            salt_count = int(width * height * 0.01 * random.random())
            y_coord = np.random.randint(0, height, [salt_count])
            x_coord = np.random.randint(0, width, [salt_count])
            image[(y_coord, x_coord)] = salt


def random_blur_or_shappen(image):
    blured = cv2.GaussianBlur(image, (5, 5), 7)
    weight = np.array(-0.5 + random.random() * 0.7, dtype=np.float32)
    image = image * (1 - weight) + blured * weight
    return image  


def random_flip(image, points):
    if random.random() > 0.5:
        image =  cv2.flip(image, 1)
        points = hp.flip_points(points, input_width)
    return image, points

def random_lines(image):
    for ch in range(0, 2):
        if random.random() < 0.05:
            step = random.randint(3, 7)
            start = random.randint(0, step - 1)
            image[start::step, :, ch] = random.randint(0, 255)

    for ch in range(0, 2):
        if random.random() < 0.05:
            step = random.randint(3, 7)
            start = random.randint(0, step - 1)
            image[:, start::step, ch] = random.randint(0, 255)

def random_dim(image):
    for ch in range(0, 3):
        image[:, :, ch] += random.randint(-10, 10)

def random_ditter(image):
    if random.random() < 0.5:
        image += np.random.normal(0, random.randint(1, 20), image.shape)


def random_multi_tone(image):
    if random.random() < 0.1:
        sep = random.randint(image.shape[1] // 4, image.shape[1] * 3 // 4)
        offset = (np.random.random([3]) - 0.5) * 30
        image[:, :sep, :] += offset
        offset = (np.random.random([3]) - 0.5) * 30
        image[:, sep:, :] += offset
    if random.random() < 0.1:
        sep = random.randint(image.shape[1] // 4, image.shape[1] * 3 // 4)
        offset = (np.random.random([3]) - 0.5) * 30
        image[:sep, :, :] += offset
        offset = (np.random.random([3]) - 0.5) * 30
        image[sep:, :, :] += offset

def random_shadow(image):
    width = image.shape[1]
    height = image.shape[0]

    ver_shadow = (np.arange(height) - (height / 2)) / (height / random.randint(1, 30))
    hor_shadow = (np.arange(width) - (width / 2)) / (width / random.randint(1, 30))

    ver_shadow = np.linspace(0, 1, height).reshape([-1, 1, 1])
    hor_shadow = np.linspace(0, 1, width).reshape([1, -1, 1])
    
    ver_shadow *= np.random.randn() * 40
    hor_shadow *= np.random.randn() * 40

    image += ver_shadow.astype(np.float32)
    image += hor_shadow.astype(np.float32)

def aug_all(image, points, bbox):
    image, points = random_geometry(image, points, bbox)
    image, points = random_flip(image, points)
    

    image = image.astype(np.float32)
    # print(image.shape)
    random_dim(image)
    random_ditter(image)
    random_salt_and_papper(image)
    image = image.astype(np.float32)
    np.clip(image, 0, 255)
    
    return image, points

def aug_eval(image, points, bbox):
    image, points = random_geometry(image, points, bbox)
    image, points = random_flip(image, points)
    image = image.astype(np.float32)
    
    return image, points
    

if __name__ == '__main__':
    data = json.load(open(os.path.join(args.coco_path, 'annotations/person_keypoints_val2017.json')))

    anno = data['annotations'][56]
    image = data['images'][0]

    image_path = os.path.join(args.coco_path, 'val2017', image['file_name'])
    src = cv2.imread(image_path)

    points = np.array(anno['keypoints']).reshape(17, -1)[:, :2]

    while True:
        dst, dst_points = random_geometry(src, points, anno['bbox'])
        coco_helper.draw_a_pose(dst, dst_points)
        cv2.imwrite('output.jpg', dst)
        print(image_path)