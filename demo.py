###conda activate test_yolov5

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

from libs.bbox3d_utils import *
from libs.Plotting import *
from model import model_mobilenet,orientation_loss

from utils import *

####### select model  ########
# select_model = 'resnet50'
# select_model ='resnet101'
# select_model = 'resnet152'
# select_model = 'vgg11'
# select_model = 'vgg16'
# select_model = 'vgg19'
# select_model = 'efficientnetb0'
# select_model = 'efficientnetb5'
select_model = 'mobilenetv2'


# Load the 3D model
bbox3d_model = tf.keras.models.load_model('mobilenetv2\\3D_box_bestweights.h5',custom_objects={'orientation_loss':orientation_loss})

# print(bbox3d_model.summary())
print('loading file ...'+select_model+'_weights.h5...!')




# Load a 2D model
bbox2d_model = YOLO('yolov8n.pt')  # load an official model
# set model parameters
bbox2d_model.overrides['conf'] = 0.9  # NMS confidence threshold
bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
bbox2d_model.overrides['classes'] = 2 ## define classes


# Load the video
video = cv2.VideoCapture('assets\\2011_10_03_drive_0034_sync_video_trimmed.mp4')

print("Process video done")
### svae results
# Get video information (frame width, height, frames per second)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
out = cv2.VideoWriter(select_model+'_output_video.mp4', fourcc, 15, (frame_width, frame_height))




frameId = 0
start_time = time.time()
fps = str()
BEV_plot = False
TracK = False

# Process each frame of the video
while True:
    frameId+=1
    success, frame = video.read()
    print(success)
    if not success:
        print("error frame")
        break

    img = frame.copy() 
    img2 = frame.copy() 
    img3 = frame.copy()
    plot3dbev = Plot3DBoxBev(P2)  

    ## process 2D and 3D boxes
    img2D, bboxes2d = process2D(bbox2d_model,img2, track=TracK)
    # print(bboxes2d)
    if len(bboxes2d) > 0:
        bboxes3D = process3D(bbox3d_model,frame,img, bboxes2d)
        # print(bboxes3D)
        if len(bboxes3D) > 0:
            for bbox_, dim, alpha, theta_ray, orient, conf,  classes, location, objID in bboxes3D:
                plot3d(img3, P2, bbox_, dim, alpha, theta_ray)

                if BEV_plot:
                    # initialize object container
                    obj = KITTIObject()
                    obj.name = str(yolo_classes[int(classes.cpu().numpy())])
                    obj.truncation = float(0.00)
                    obj.occlusion = int(-1)
                    obj.xmin, obj.ymin, obj.xmax, obj.ymax = int(bbox_[0]), int(bbox_[1]), int(bbox_[2]), int(bbox_[3])

                    obj.alpha = recover_angle(orient, conf, bin_size)
                    obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
                    obj.rot_global, rot_local = compute_orientaion(P2, obj)
                    obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

                    # plot 3d BEV bbox
                    rot_y = alpha + theta_ray
                    plot3dbev.plot(img=img3, class_object=obj.name.lower(), 
                      bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                      dim=[obj.h, obj.w, obj.l], loc=[obj.tx, obj.ty, obj.tz], rot_y=rot_y, objId=[objID] if not isinstance(objID, list) else objID)


    if BEV_plot:
        plot3dbev.plot(img=img3)
        img3 = plot3dbev.show_result()


    # Calculate the current time in seconds
    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if frameId % 20 == 0:  # Calculate FPS every 10 frames
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_current = frameId / elapsed_time
        fps =  f'FPS: {fps_current:.2f}'
        print(f'Frame: {frameId}, FPS: {fps_current:.2f}')
    cv2.putText(img3, select_model+' '+fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)


    # Display the frame
    cv2.imshow("2D", img2)
    cv2.imshow("3D", img3)
    out.write(img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture featuresect
video.release()
cv2.destroyAllWindows()
