import json
import os
import random
from collections import defaultdict
import numpy as np

import tensorflow as tf

from params import args
import coco_helper as hp

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_record(data, image_dir, record_filename):
    writer = tf.io.TFRecordWriter(record_filename)
    annos = data['annotations']

    image_file_names = {image['id']:image['file_name'] for image in data['images']}
    result = []
    
    #any valid annotation must include these body parts
    must_have = np.array([
        hp.NOSE * 3, hp.NOSE * 3 + 1,
        hp.LEFT_WRIST * 3, hp.LEFT_WRIST * 3 + 1,
        hp.RIGHT_WRIST * 3, hp.RIGHT_WRIST * 3 + 1,
        hp.LEFT_ANKLE * 3, hp.LEFT_ANKLE * 3 + 1,
        hp.RIGHT_ANKLE * 3, hp.RIGHT_ANKLE * 3 + 1,
        ])

    for anno in annos:
        keypoints = np.array(anno['keypoints'])
        if keypoints[must_have].all():
            points = np.array(anno['keypoints'], dtype=np.float32)
            points = points.reshape(17, -1)[:, :2]
            valid = points.all(axis=1).astype(np.int64)
            bbox = np.array(anno['bbox'], dtype=np.float32)
            file_name = image_file_names[anno['image_id']]
            path = os.path.join(args.coco_path, image_dir, file_name)
            file = open(path, 'rb')
            feat_dict = {}
            feat_dict['image'] = bytes_feature([file.read()])
            feat_dict['points'] = floats_feature(points.reshape(-1))
            feat_dict['valid'] = int64_feature(valid)
            feat_dict['bbox'] = floats_feature(bbox)
            example = tf.train.Example(features=tf.train.Features(feature=feat_dict))
            writer.write(example.SerializeToString())

    writer.close()

#load coco json
print('load train json ...')
train_data = json.load(open(os.path.join(args.coco_path, 'annotations/person_keypoints_train2017.json')))
print('load val json ...')
val_data = json.load(open(os.path.join(args.coco_path, 'annotations/person_keypoints_val2017.json')))

os.makedirs(args.records_dir, exist_ok=True)
print('write train tfrecord ...')
write_record(train_data, 'train2017', os.path.join(args.records_dir, 'train.tfrecord'))
print('write val tfrecord ...')
write_record(val_data, 'val2017', os.path.join(args.records_dir, 'val.tfrecord'))

print('done.')