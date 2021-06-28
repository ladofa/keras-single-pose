import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.python.ops.parsing_ops import FixedLenFeature

from params import args
import augments



def read_tfrecord(example):
    tfrecord_format = (
        {
            'image':tf.io.FixedLenFeature([], tf.string),
            'points':tf.io.FixedLenFeature([34], tf.float32),
            'valid':tf.io.FixedLenFeature([17], tf.int64),
            'bbox':tf.io.FixedLenFeature([4], tf.float32),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    points = tf.reshape(example['points'], (17, -1))
    valid = example['valid'] 
    bbox = example['bbox']
    return image, points, valid, bbox

dataset_train = tf.data.TFRecordDataset(os.path.join(args.records_dir, 'train.tfrecord'))
dataset_valid = tf.data.TFRecordDataset(os.path.join(args.records_dir, 'val.tfrecord'))
dataset_train = dataset_train.map(read_tfrecord)
dataset_valid = dataset_valid.map(read_tfrecord)


if __name__ == '__main__':
    for image, points, valid, bbox in dataset_valid.take(5):
        cv2.imwrite('output.jpg', image.numpy())
        print(points)
        print(valid)
        print(bbox)
        print('checkpoint..')

input_width = args.input_width
input_height =args.input_height
output_width = args.output_width
output_height = args.output_height
bw = input_width / output_width
bh = input_height / output_height
bw2 = bw / 2
bh2 = bh / 2

center_x = []
center_y = []
for y in range(output_height):
    for x in range(output_width):
        center_x.append(x * bw + bw2)
        center_y.append(y * bh + bh2)
center_x = tf.constant(center_x, dtype=tf.float32)
center_y = tf.constant(center_y, dtype=tf.float32)

@tf.function
def encode_points(points):
    i_x = tf.clip_by_value(points[..., 0] // bw, 0, output_width - 1)
    i_y= tf.clip_by_value(points[..., 1] // bh, 0, output_height - 1)
    points_label = tf.cast(i_y * output_width + i_x, tf.int32)
    off_x = (points[..., 0] - tf.gather(center_x, points_label)) / bw2
    off_y = (points[..., 1] - tf.gather(center_y, points_label)) / bh2
    off_x = off_x[..., None]
    off_y = off_y[..., None]
    offset = tf.concat([off_x, off_y], axis=-1)

    return points_label, offset

@tf.function
def decode_points(points_label, offset):
    cx = tf.gather(center_x, points_label)
    cy = tf.gather(center_y, points_label)
    off_x = offset[..., 0]
    off_y = offset[..., 1]
    x = cx + off_x * bw2
    y = cy + off_y * bh2
    x = x[..., None]
    y = y[..., None]
    points = tf.concat([x, y], axis=-1)
    return points
    

def aug_train(image, points, valid, bbox):
    image, points = tf.numpy_function(augments.aug_all, [image, points, bbox], (tf.float32, tf.float32))
    points_label, offset = encode_points(points)
    return image, (points_label, offset, valid)

def aug_valid(image, points, valid, bbox):
    image, points = tf.numpy_function(augments.aug_eval, [image, points, bbox], (tf.float32, tf.float32))
    points_label, offset = encode_points(points)
    return image, (points_label, offset, valid)


dt = dataset_train.map(aug_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dt = dt.shuffle(args.buffer_size).batch(args.batch_size).prefetch(1)
dv = dataset_valid.map(aug_valid, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dv = dv.batch(args.batch_size).prefetch(1)

if __name__ == '__main__':
    for image, (points, valid, bbox) in dt.take(1):
        cv2.imwrite('output.jpg', image[0].numpy())
        print(points)
        print(valid)
        print(bbox)
        print('checkpoint..')