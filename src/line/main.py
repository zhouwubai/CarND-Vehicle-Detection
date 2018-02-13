from moviepy.editor import VideoFileClip

from line.models import LineDetector
from line.transform import Calibrator, Unwarper
from constants import ROOT


if __name__ == '__main__':

    cal_input = ROOT + 'models/calibration.pkl'
    calibrator = Calibrator(path=cal_input)
    unwarp_input = ROOT + 'models/unwarp.pkl'
    unwarper = Unwarper(path=unwarp_input)

    video_input = ROOT + 'video/project_video.mp4'
    video_output = ROOT + 'video/project_video_labeled_line.mp4'
    clip1 = VideoFileClip(video_input)
    detector = LineDetector(calibrator=calibrator, unwarper=unwarper,
                            keep_n=10, alpha=0.6)
    # NOTE: this function expects    color images!!
    white_clip = clip1.fl_image(detector.process)
    white_clip.write_videofile(video_output, audio=False)


