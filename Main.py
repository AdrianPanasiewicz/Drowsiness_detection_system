import pathlib
import time

from fastai.vision.all import *
import cv2

from Workspace.Back_End.image_processsor import ImageProcessor
from Workspace.Back_End.file_loader import FileLoader

# A work-around for the error that PosixPath cannot be instantiated on your system
# More about this issue on https://github.com/ultralytics/yolov5/issues/10240
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Załadowanie klasy ImageProcessor do przetwarzania obrazu
image_processor = ImageProcessor()
crop_size = (224,224)

file_loader = FileLoader()


# Load model
emotion_model = load_learner(
    r'C:\Users\adria\PycharmProjects\Real-time-drowsiness-and-emotion-detection\Workspace\Resources\Models\Emotion_Model.pkl')

# Initialise camera for video capture
camera = cv2.VideoCapture(0)

while true:
    start_tick = time.process_time()
    ret, frame = camera.read()
    processed_frame = image_processor.preprocess_image(frame, *crop_size)
    output = emotion_model.predict(processed_frame)
    cv2.putText(processed_frame, output[0], (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
    cv2.imshow('Drowsiness detection', processed_frame)
    stop_tick = time.process_time()
    print(stop_tick - start_tick)
    k = cv2.waitKey(5)


camera.release()
cv2.destroyAllWindows()

# # Main loop
# while ret:
#     # Read a frame from video and predict drowsiness
#     ret, frame = camera.read()
#     output = drowsiness_model.predict(frame)
#
#     print(output)
#
#     # Display the results on the frame
#     cv2.putText(frame, output[0], (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 3, [17, 163, 252], 4)
#     cv2.imshow('Drowsiness detection', frame)
#     k = cv2.waitKey(10)
#
# camera.release()
# cv2.destroyAllWindows()
