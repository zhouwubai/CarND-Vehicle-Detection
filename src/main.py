import pickle
import cv2
from moviepy.editor import VideoFileClip

from line.models import LineDetector
from car.models import ModelType
from car.piper import CarDectorPiper
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

    name = ModelType.LinearSVC
    model_path = ROOT + 'models/car_model_{}.pkl'.format(name)
    print('Loading model in {}...'.format(model_path))
    car_detector = pickle.load(open(model_path, 'rb'))
    print(car_detector.print_params())

    keep_n = 6
    threshold = 10
    draw_heatmap = True
    piper = CarDectorPiper(car_detector,
                           keep_n=keep_n,
                           threshold=threshold,
                           draw_heatmap=draw_heatmap,
                           standalone=False)

    oracle_detector = OracleDetector(line_detector=line_detector,
                                     car_detector=piper)

    video_input = ROOT + 'video/project_video.mp4'
    video_output = ROOT + 'video/project_video_labeled_line_car.mp4'
    clip1 = VideoFileClip(video_input)

    # NOTE: this function expects    color images!!
    white_clip = clip1.fl_image(oracle_detector.process)
    white_clip.write_videofile(video_output, audio=False)


