import argparse
import cv2
import os
from image_detector import ImageDetector

TRAINED_MODEL_DEFAULT = 'trained_model/snapshot_model_20180404.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default=TRAINED_MODEL_DEFAULT)
    parser.add_argument('image')
    args = parser.parse_args()

    imageDetector = ImageDetector(args.pretrained_model)
    return_list = imageDetector.detect(args.image, args.gpu)

    img = cv2.imread(args.image)
    for (_, x, y, w, h) in return_list:
        # print(x, y, w, h)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

    cv2.imwrite(os.path.splitext(args.image)[0] + "_predicted" + os.path.splitext(args.image)[1], img)


if __name__ == '__main__':
    main()
