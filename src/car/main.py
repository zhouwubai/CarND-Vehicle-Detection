import pickle

from car.models import ModelType
from car.piper import CarDectorPiper
from constants import ROOT

from moviepy.editor import VideoFileClip


if __name__ == '__main__':

    name = ModelType.LinearSVC
    model_path = ROOT + 'models/car_model_{}.pkl'.format(name)
    print('Loading model in {}...'.format(model_path))
    detector = pickle.load(open(model_path, 'rb'))
    print(detector.print_params())

    keep_n = 6
    threshold = 10
    draw_heatmap = True
    piper = CarDectorPiper(detector,
                           keep_n=keep_n,
                           threshold=threshold,
                           draw_heatmap=draw_heatmap)

    video_input = ROOT + 'video/test_video.mp4'
    video_output = ROOT + 'video/test_video_labeled_car.mp4'
    clip1 = VideoFileClip(video_input)
    white_clip = clip1.fl_image(piper.process)
    white_clip.write_videofile(video_output, audio=False)


