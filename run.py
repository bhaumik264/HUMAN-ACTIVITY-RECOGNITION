import cv2
import pickle
import argparse
import importlib
from extractor import PoseExtractor
# import numpy as np

parser = argparse.ArgumentParser(description='Run inference on webcam video')
parser.add_argument('--config', type=str, default='config',
                   help="config.py file inside config/ directory, default: 'config.py'")
args = parser.parse_args()
config = importlib.import_module('config.' + args.config)
 # load the model
model = pickle.load(open(config.classifier_model, 'rb'))
pose_extractor = PoseExtractor()
stream = 0
# stream = ‘/path/to/video/file’
camera = cv2.VideoCapture(stream)
while(camera.isOpened()):
   ret, frame = camera.read()
   if ret == True:
       frame = cv2.flip(frame,1)
       pose_data = pose_extractor.extract([frame])
       activity = model.predict(pose_data.reshape(1, -1))
       cv2.putText(frame, activity[0],(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0, 0),2         ,cv2.LINE_AA)
       cv2.imshow(‘Human Activity Recognition’, frame)
       print(activity[0])
      if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   else:
       break
 camera.release()
 cv2.destroyAllWindows()
