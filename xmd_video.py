

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'video.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


MODEL_NAME2 = 'inference_graph2'
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT2 = os.path.join(CWD_PATH,MODEL_NAME2,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS2 = os.path.join(CWD_PATH,'training2','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO2 = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES2 = 43

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map2 = label_map_util.load_labelmap(PATH_TO_LABELS2)
categories2= label_map_util.convert_label_map_to_categories(label_map2, max_num_classes=NUM_CLASSES2, use_display_name=True)
category_index2 = label_map_util.create_category_index(categories2)

# Load the Tensorflow model into memory.
detection_graph2 = tf.Graph()
with detection_graph2.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT2, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess2 = tf.Session(graph=detection_graph2)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor2 = detection_graph2.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes2 = detection_graph2.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores2 = detection_graph2.get_tensor_by_name('detection_scores:0')
detection_classes2 = detection_graph2.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections2 = detection_graph2.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

       # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
       # i.e. a single-column array, where each item in the column has the pixel RGB value
       ret, frame = video.read()
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       frame_expanded = np.expand_dims(frame_rgb, axis=0)

       # Perform the actual detection by running the model with the image as input
       (boxes, scores, classes, num) = sess.run(
       [detection_boxes, detection_scores, detection_classes, num_detections],
       feed_dict={image_tensor: frame_expanded})
		
       coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=3,
                        min_score_thresh=0.9)
       if  coordinates:
        x=0
        #for coordinate in coordinates:
        #(y1, y2, x1, x2,acc) = coordinate
        y1 = coordinates[0][0]
        y2 = coordinates[0][1]
        x1 = coordinates[0][2]
        x2 = coordinates[0][3]
        height = y2-y1
        width = x2-x1
        height=height+14
        width=width+14
        crop = frame[y1:y1+height, x1:x1+width]
	    #cv2.imshow('Object detector', crop)
        path=os.path.join(CWD_PATH,'opencv.png')
        cv2.imwrite(path,crop);
	    #tf.reset_default_graph()
        image2=cv2.imread(os.path.join(CWD_PATH,'opencv.png'))
		#image_rgb2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#image_expanded2 = np.expand_dims(image_rgb2, axis=0)
        image_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        
        (boxes2, scores2, classes2, num2) = sess2.run(
        [detection_boxes2, detection_scores2, detection_classes2, num_detections2],
        feed_dict={image_tensor2: image_expanded})
		# Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes[0, x:x+4]),
        np.squeeze(classes2).astype(np.int32),
        np.squeeze(scores2),
        category_index2,
        use_normalized_coordinates=True,
		line_thickness=8,
	    min_score_thresh=0.90)
        x+=1
		# All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        

        # Press 'q' to quit
       if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
