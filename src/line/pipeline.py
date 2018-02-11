from moviepy.editor import VideoFileClip
from line.models import Road
from constants import ROOT

if __name__ == '__main__':

    video_input = ROOT + 'video/project_video.mp4'
    video_output = ROOT + 'video/project_video_labeled_line.mp4'
    clip1 = VideoFileClip(video_input)
    road = Road(keep_n=10, alpha=0.2)
    # NOTE: this function expects    color images!!
    white_clip = clip1.fl_image(road.process_image)
    white_clip.write_videofile(video_output, audio=False)


