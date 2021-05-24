import argparse
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
    print(return_list)


if __name__ == '__main__':
    main()
