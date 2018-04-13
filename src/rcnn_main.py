from moviepy.editor import VideoFileClip

from line.models import LineDetector
from faster_rcnn.piper import FasterRCNNCarDetector
from line.transform import Calibrator, Unwarper
from constants import ROOT
from utils import draw_boxes


class OracleDetector(object):
    def __init__(self, line_detector, car_detector):
        self.line_detector = line_detector
        self.car_detector = car_detector

    def process(self, image):
        line_img = self.line_detector.process(image)
        alive_box = self.car_detector.process(image)
        result = draw_boxes(line_img, alive_box)

        return result


if __name__ == '__main__':

    cal_input = ROOT + 'models/calibration.pkl'
    calibrator = Calibrator(path=cal_input)
    unwarp_input = ROOT + 'models/unwarp.pkl'
    unwarper = Unwarper(path=unwarp_input)
    line_detector = LineDetector(calibrator=calibrator,
                                 unwarper=unwarper,
                                 keep_n=10, alpha=0.6)

    model_path = ROOT +\
        'data/faster_rcnn/checkpoints/fasterrcnn_caffe_pretrain.pth'
    print('Loading model in {}...'.format(model_path))
    car_detector = FasterRCNNCarDetector(model_path)

    oracle_detector = OracleDetector(line_detector=line_detector,
                                     car_detector=car_detector)

    video_input = ROOT + 'video/project_video.mp4'
    video_output = ROOT + 'video/project_video_labeled_fasterrcnn.mp4'
    clip1 = VideoFileClip(video_input)

    # NOTE: this function expects    color images!!
    white_clip = clip1.fl_image(oracle_detector.process)
    white_clip.write_videofile(video_output, audio=False)


