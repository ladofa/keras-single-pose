# 각종 파라미터를 모아놓음

import argparse


parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')

parser.add_argument('--input-width', type=int, default=224)
parser.add_argument('--input-height', type=int, default=448)
parser.add_argument('--output-width', type=int, default=7)
parser.add_argument('--output-height', type=int, default=14)
parser.add_argument('--coco-path', type=str, default="/home/ubuntu/datasets/coco")
parser.add_argument('--records_dir', type=str, default="records")

#################################################
#이하 컴퓨터, 테스트마다 달라짐
# parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--buffer-size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=10)





args = parser.parse_args()

if __name__ == '__main__':
    print(args)