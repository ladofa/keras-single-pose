import os
import json
import numpy as np
import cv2
from collections import defaultdict

from params import args


NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


connections = [
    (LEFT_EAR, LEFT_EYE ),
    (LEFT_EYE, NOSE ),
    (NOSE, RIGHT_EYE ),
    (RIGHT_EYE, RIGHT_EAR ),

    (LEFT_ELBOW, LEFT_WRIST ),
    (LEFT_ELBOW, LEFT_SHOULDER ),
    (LEFT_HIP, LEFT_SHOULDER ),
    (LEFT_HIP, LEFT_KNEE ),
    (LEFT_KNEE, LEFT_ANKLE ),

    (RIGHT_HIP, RIGHT_SHOULDER ),
    (RIGHT_ELBOW, RIGHT_SHOULDER ),
    (RIGHT_ELBOW, RIGHT_WRIST ),
    (RIGHT_HIP, RIGHT_KNEE ),
    (RIGHT_KNEE, RIGHT_ANKLE ),

    (LEFT_SHOULDER, RIGHT_SHOULDER ),
    (LEFT_HIP, RIGHT_HIP)
]

colors = [
    (0, 255, 255),
    (0, 255, 255),
    (0, 255, 255),
    (0, 255, 255),
    
    (0, 0, 255),
    (0, 64, 255),
    (0, 128, 255),
    (64, 64, 255),
    (128, 0, 255),

    (255, 0, 0),
    (255, 64, 0),
    (255, 128, 0),
    (255, 64, 64),
    (255, 0, 128),

    (128, 255, 0),
    (0, 255, 128)
]


def draw_a_pose(image, points):
    points = np.array(points, dtype=np.int32).reshape(17, -1)
    points = points[:, :2]

    for con, color in zip(connections, colors):
        p0 = points[con[0]]
        p1 = points[con[1]]
        if (p0[0] == 0 and p0[1] == 0) or (p1[0] == 0 and p1[1] == 0):
            continue
        cv2.line(image, p0, p1, color, thickness=2)

#just visualize data
if __name__ == '__main__':
    data = json.load(open(os.path.join(args.coco_path, 'annotations/person_keypoints_val2017.json')))

    annos = data['annotations']
    image_to_anno = defaultdict(list)
    for anno in annos:
        image_to_anno[anno['image_id']].append(anno)
    
    for image in data['images']:
        image_path = os.path.join(args.coco_path, 'val2017', image['file_name'])
        src = cv2.imread(image_path)
        
        for anno in image_to_anno[image['id']]:
            draw_a_pose(src, anno['keypoints'])
        # cv2.imshow('src', src)
        # cv2.waitKey(0)
        cv2.imwrite('output.jpg', src)
        print(image_path)
        
    


def flip_points(points, width):
    points[:, 0] = width - points[:, 0]
    points = points[[
        NOSE,
        RIGHT_EYE,
        LEFT_EYE,
        RIGHT_EAR,
        LEFT_EAR,
        RIGHT_SHOULDER,
        LEFT_SHOULDER,
        RIGHT_ELBOW,
        LEFT_ELBOW,
        RIGHT_WRIST,
        LEFT_WRIST ,
        RIGHT_HIP ,
        LEFT_HIP ,
        RIGHT_KNEE ,
        LEFT_KNEE ,
        RIGHT_ANKLE ,
        LEFT_ANKLE]]
    return points.astype(np.float32)