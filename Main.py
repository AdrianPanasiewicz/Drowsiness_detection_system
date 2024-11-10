import cv2
from fastai.vision.all import *
import pathlib
from Workspace.Back_End.ImageProcessor import ImageProcessor

# A work-around for the error that PosixPath cannot be instantiated on your system
# More about this issue on https://github.com/ultralytics/yolov5/issues/10240
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Załadowanie klasy ImageProcessor do przetwarzania obrazu
image_processor = ImageProcessor()
crop_size = (224,224)

# Load model
# drowsiness_model = load_learner(
#     r'C:\Users\adria\PycharmProjects\Real-time-drowsiness-and-emotion-detection\Models\Emotion_Model.pkl')

# Initialise camera for video capture
camera = cv2.VideoCapture(0)

while true:
    ret, frame = camera.read()
    processed_frame = image_processor.preprocess_image(frame, *crop_size)
    cv2.imshow('Drowsiness detection', processed_frame)
    k = cv2.waitKey(20)

input("Press enter to continue...")

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
