import cv2
from fastai.vision.all import *

camera = cv2.VideoCapture(0)
# drowsiness_detector = load_learner()

ret, frame = camera.read()

while ret:
    ret, frame = camera.read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(25)

camera.release()
cv2.destroyAllWindows()
