import cv2
from fastai.vision.all import *
import pathlib

# A work-around for the error that PosixPath cannot be instantiated on your system
# More about this issue on https://github.com/ultralytics/yolov5/issues/10240
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load model
drowsiness_model = load_learner(r'C:\Users\adria\PycharmProjects\Real-time-drowsiness-and-emotion-detection\Models\Drowsiness_model.pkl')

# Initialise camera for video capture
camera = cv2.VideoCapture(0)
ret, frame = camera.read()

while ret:
    ret, frame = camera.read()
    cv2.imshow('Drowsiness detection', frame)
    output = drowsiness_model.predict(frame)
    print(output)
    k = cv2.waitKey(10)

camera.release()
cv2.destroyAllWindows()
