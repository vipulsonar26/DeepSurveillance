# USAGE
# python model.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --video_file trained/Back To School Bags.mp4  --in 	

from imutils.video import VideoStream
from imutils.video import FPS

import datetime
import sys
import numpy as np
import argparse
import imutils
import time
import cv2
import math
import os
from os.path import isfile, join

sys.path.append("/home/lenovo/caffe/python")
import caffe
from caffe.proto import caffe_pb2

caffe.set_mode_cpu() 


tot=0
count =0

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")

ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")

ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
	
ap.add_argument("-t", "--video_file", required=True,
	help="minimum probability to filter weak detections")

ap.add_argument("-r", "--in", required=True,
	help="")

args = vars(ap.parse_args())


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "Unknown_object", "bicycle", "bird", "Object",
	"bottle", "bus", "car", "cat", "chair", "object", "diningtable",
	"dog", "horse", "motorbike", "Human", "pottedplant", "object",
	"sofa", "Vehicle", "tvmonitor"]
	
CLASSES_trained = ["bag","laptop"]
	
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/lenovo/Project/DeepSurveillance-master/real-time-object-detection/trained/baglaptop_mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net2 = caffe.Net('/home/lenovo/Project/DeepSurveillance-master/real-time-object-detection/trained/deploy.prototxt.txt',
                '/home/lenovo/Project/DeepSurveillance-master/real-time-object-detection/trained/caffe_alexnet_train_iter_450000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

preds = []

path = "/home/lenovo/Project/DeepSurveillance-master/real-time-object-detection/Frames/" + args['video_file'] + '_' + str(datetime.datetime.now())
path2= path + '/'+"req_obj"
os.mkdir( path, 0755);
os.mkdir(path2,0755);

#os.system("python /home/lenovo/Project/DeepSurveillance-master/real-time-object-detection/test_h.py " +args['prototxt'] )
print(net)
fn=sys.argv[0]


cap=cv2.VideoCapture(args['video_file'])

frameRate = cap.get(5)
temp=0

while (cap.isOpened()):
	ret,frame = cap.read()
	frame1 = frame

	tot+=1
	if ret== True:
		frameId = cap.get(1)
		#print (frameId)
		frame = imutils.resize(frame, width=800)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, 		        
		                                                           (300,300),127.5)
	# pass the blob through the network and obtain the detections and
	# predictions
		net.setInput(blob)
		detections = net.forward()
		#print (detections)
	
		for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		
		# the prediction
			confidence =detections[0,0,i,2]
			#print(i)          it is for object

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
			if confidence > args["confidence"]:
				
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
				
				idx = int(detections[0, 0, i, 1])
				
				#print(COLORS[idx])
				t=confidence * 100
				if (idx!=temp and (t < 30 or t > 97)) :
					if COLORS[idx].any():
						filename = path
												
						if CLASSES[idx] == args['in']:
							cv2.imwrite(path2 +'/'+ "img%d.jpg" % count,frame1)
							
						else :	
						#Trained Used here
							cv2.imwrite(filename +'/'+"frame%d.jpg" % count,frame1)
							if args["in"] != "none": 
								img = cv2.resize(frame, (227,227))
								net2.blobs['data'].data[...] = transformer.preprocess('data', img)
								out = net2.forward()
								pred_probas = out['prob']
								#print(pred_probas)
								preds = preds + [pred_probas.argmax()]
							#print(label)
								if CLASSES_trained[pred_probas.argmax()] == args["in"]:
									cv2.imwrite(path2 +'/'+ "img%d.jpg" % count,frame)			
						count +=1
				temp=idx		
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else :
		break


#generate summary

pathOut = path2+'/'+'A_'+'test.mp4'
if count < 200:
	fps = 24.0
else:
	fps = 30.0
if args['in'] == 'none':
	pathIn=path+'/'
else:
	pathIn =path2 + '/'
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
files.sort(key=lambda f: int(filter(str.isdigit, f)))
 
for i in range(len(files)):
	filename=pathIn + files[i]
        #reading each files
	img=cv2.imread(filename)
	height, width, layers = img.shape
	size = (width,height)
#    print(filename)
        #inserting the frames into an image array	
	frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(frame_array)):
      # writing to a image array
    out.write(frame_array[i])
out.release()
 

print(count)
print(tot)
cap.release()
cv2.destroyAllWindows()
