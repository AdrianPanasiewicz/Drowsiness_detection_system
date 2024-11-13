import pathlib
import time

from fastai.vision.all import *
import mediapipe as mp
import cv2


from Workspace import *

if __name__ == "__main__":
    # A work-around for the error that PosixPath cannot be instantiated on your system
    # More about this issue on https://github.com/ultralytics/yolov5/issues/10240
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    # Załadowanie klasy ImageProcessor do przetwarzania obrazu
    image_processor = ImageProcessor()
    crop_size = (224,224)

    model_loader = ModelLoader()
    models = model_loader.load_models()
    # Initialise camera for video capture

    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError

    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    past_tick = 0

    while True:
        fps = Utils.calculate_fps()

        ret, frame = camera.read()
        processed_frame = image_processor.preprocess_image(frame, *crop_size)

        # output = models["Emotion_model"].predict(processed_frame)
        # cv2.putText(processed_frame, fps, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)


        cv2.imshow('Emotion detection', processed_frame)
        stop_tick = time.process_time()
        print(fps)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


