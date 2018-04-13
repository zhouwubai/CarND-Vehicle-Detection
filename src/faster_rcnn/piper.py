import numpy as np
import torch as t
from faster_rcnn.ctorch.model.faster_rcnn_vgg16 import FasterRCNNVGG16

from faster_rcnn.ctorch.trainer import FasterRCNNTrainer


CAR_CLASS_ID = 6


class FasterRCNNCarDetector(object):

    def __init__(self, model_path):
        faster_rcnn = FasterRCNNVGG16()
        trainer = FasterRCNNTrainer(faster_rcnn).cuda()
        self.trainer = trainer.load(model_path)

    def process(self, img):
        """
        args:
            img: np.ndarray
            All images are in HWC and RGB format,
            need to transpose to CHW
            and the range of their value is :math:`[0, 255]`.
        """
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = t.from_numpy(img)[None]
        # boxes return here in the format of (y_min, x_min, y_max, x_min)
        # but expect
        _bboxes, _labels, _scores =\
            self.trainer.faster_rcnn.predict(img, visualize=True)
        bboxes, labels = _bboxes[0], _labels[0]
        car_idx = np.where(labels == CAR_CLASS_ID)
        car_bboxes = bboxes[car_idx]
        car_bboxes = self._transform_bboxes(car_bboxes)
        return car_bboxes

    def _transform_bboxes(self, bboxes):
        """
        transform bboxes (y_min, x_min, y_max, x_max)
        from neural network to [(x_min, y_min), (x_max, y_max)]
        """
        results = []
        for b in bboxes:
            results.append(((b[1], b[0]), (b[3], b[2])))
        return results
