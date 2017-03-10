import cv2
import os
import argparse
import math

from moviepy.editor import *

from lane_lib.debug import *
from lane_lib import calib
from lane_lib import detector
from lane_lib import masking
from lane_lib import perspective

if __name__ == "__main__":

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Advanced Lane Lines')

    # Example: python lanelines.py -i 'project_video.mp4' -o 'annotated_project_video.mp4'
    parser.add_argument('-i', '--inputvideo', type=str, 
                        default='project_video.mp4',
                        help='Input driving video (recommended .mp4 file)')
    parser.add_argument('-o', '--outputvideo', type=str, 
                        default='annotated_project_video.mp4',
                        help='Output video file')
    args = parser.parse_args()

    if not os.path.exists(args.inputvideo):
        raise ValueError('Input path not found.')

    # Load road video
    video = VideoFileClip(args.inputvideo)

    # Instrinsic camera calibration
    undistort = calib.CameraCalibrate()

    # Perspective transformation
    perspective = perspective.PerspectiveTransform(video.size[::-1])

    # Initialise lane line detector
    lane_detector = detector.LaneLinesDetector(undistort.mtx, undistort.dist, 
                                               perspective.M, perspective.Minv)

    # import pdb;pdb.set_trace()
    
    clip = []
    # Iterate though input video stream 
    for img in video.iter_frames(progress_bar = True):

        # Find lane lines and annotate input frame
        lane_detector.draw(img)

        # Get the result of lane lines detector
        outimg = lane_detector.result_img

        # cv2.imshow('Frame', outimg)
        # cv2.waitKey(1)
        
        # Convert OpenCV colour space to RGB before writing to file
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        clip += [outimg]
    
    # Write clips to output video file
    outvideo = ImageSequenceClip(clip, fps=30)
    outvideo.write_videofile(args.outputvideo)
