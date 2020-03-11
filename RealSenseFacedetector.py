#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import pdb
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from imutils import face_utils
import dlib

def imresize(img):
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input==", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
filename = askopenfilename(title="Select RealSense Data File")

args.input = filename
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream 
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 30)

    # Start streaming from file
    profile = pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("RealSesne Data", cv2.WINDOW_AUTOSIZE)
 
    # Create colorizer object
    colorizer = rs.colorizer()
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    # Streaming loop
   # while True:
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    start_num = 1
    end_num = 5000
    n = 0
    while (playback.current_status() == rs.playback_status.playing) and (n<(end_num+1)):
        n += 1
        time.sleep(0.033) # to replicate 30 fps
        
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        if (n > start_num-1) and (n < (end_num+1)):
            print(n)
            # Get depth frame
            color_frame = frames.get_color_frame()
            infrared_frame = frames.get_infrared_frame()

            # Colorize depth frame to jet colormap
            #depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            infrared_image = np.asanyarray(infrared_frame.get_data())
            
            color_image = imresize(color_image)
            gray_image = imresize(gray_image)
            infrared_image = imresize(infrared_image)

            rects = detector(gray_image, 0)
            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                    shape = predictor(gray_image, rect)
                    shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
                    for (x, y) in shape:
                        cv2.circle(color_image, (x, y), 2, (0, 255, 0), -1)


            # image = np.hstack((gray_image, infrared_image))

            # Render image in opencv window
            
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense',images)
            cv2.imshow("RealSesne Data", color_image)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break

finally:
    pass


