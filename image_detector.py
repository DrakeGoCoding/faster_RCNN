import matplotlib.pyplot as plot
import os
import chainer

import download_model

from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox

TRAINED_MODEL_OLD = 'trained_model/snapshot_model.npz'
MODEL_URL_OLD = 'http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model.npz'

# A pretrained model for the higher version of chainercv.
TRAINED_MODEL_DEFAULT = 'trained_model/snapshot_model_20180404.npz'
MODEL_URL = 'http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model_20180404.npz'


class ImageDetector:
    def __init__(self, model=None):
        if model is not None:
            self.model = FasterRCNNVGG16(n_fg_class=1, pretrained_model=model)
        else:
            self.prepare()

    def prepare(self):
        chainer.config.train = False

        # Workaround for the newer version of chainercv (maybe v0.7.0 and higher)
        if not os.path.exists(TRAINED_MODEL_DEFAULT):
            download_model.download_model(MODEL_URL, TRAINED_MODEL_DEFAULT)
        try:
            self.model = FasterRCNNVGG16(
                n_fg_class=1,
                pretrained_model=TRAINED_MODEL_DEFAULT)

        except KeyError:
            if not os.path.exists(TRAINED_MODEL_OLD):
                download_model.download_model(MODEL_URL_OLD, TRAINED_MODEL_OLD)
                self.model = FasterRCNNVGG16(
                    n_fg_class=1,
                    pretrained_model=TRAINED_MODEL_OLD)
            else:
                raise

    def detect(self, image, gpu=-1):
        if gpu >= 0:
            self.model.to_gpu(gpu)
            chainer.cuda.get_device(gpu).use()

        img = utils.read_image(image, color=True)
        bboxes, labels, scores = self.model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        # vis_bbox(img, bbox, label, score, label_names=('face',))
        # plot.show()

        return_list = []
        for b, s in zip(bbox, score):
            y_min, x_min, y_max, x_max = b
            w = x_max - x_min
            h = y_max - y_min
            return_list.append([s, x_min, y_min, w, h])

        return return_list

