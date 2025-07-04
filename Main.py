from Workspace import *
import threading
import sys
import pathlib


def main():
    cfg = Config()
    pathlib.PosixPath = Utils.fix_pathlib()

    image_processor = ImageProcessor()
    coordinates_parser = CoordinatesParser()
    classifier = RandomForest(activation_certainty=0.5,
                              prediction_memory_size=50)

    perclos_finder = PerclosFinder(
        cfg.perclos_threshold)
    yawn_finder = YawnFinder(cfg.yawn_threshold)
    face_angle_finder = angle_finder.AngleFinder()

    if cfg.mode == 'camera':
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise IOError("Camera access failed")

        sql_saver = DataSaver(cfg.results_name)
        gui = GUI()

        camera_mode = CameraMode(
            camera, image_processor, coordinates_parser,
            sql_saver,
            perclos_finder, yawn_finder, face_angle_finder,
            classifier, gui
        )

        thread = threading.Thread(target=camera_mode.run,
                                  daemon=True)
        thread.start()
        gui.start()
        camera.release()

    elif cfg.mode == 'image':
        folders = [
            (cfg.training_folder / "train" / "images",
             DataSaver(cfg.training_name)),
            (cfg.training_folder / "valid" / "images",
             DataSaver(cfg.validating_name)),
            (cfg.training_folder / "test" / "images",
             DataSaver(cfg.testing_name))
        ]

        image_mode = ImageMode(
            image_processor, None, perclos_finder,
            yawn_finder, face_angle_finder, classifier
        )

        for folder, saver in folders:
            image_mode.sql_saver = saver
            image_mode.process_folder(folder)

        print(f"Processing complete. Results saved")

    elif cfg.mode == 'video':
        if cfg.processing_mode in ["training",
                                   "evaluation"]:
            processor = VideoProcessor(
                image_processor, perclos_finder,
                yawn_finder, face_angle_finder, classifier
            )

            folder = cfg.training_folder if cfg.processing_mode == "training" else cfg.validation_folder
            output = cfg.output_folder / (
                "Training" if cfg.processing_mode == "training" else "Validation")
            output.mkdir(parents=True, exist_ok=True)

            videos = [f for f in folder.rglob('*') if
                      f.is_file() and f.suffix in ['.mp4',
                                                   '.avi']]

            for video in videos:
                print(f"Processing: {video.name}")
                save_path = processor.process_video(video,
                                                    output,
                                                    cfg.processing_mode)
                print(f"Saved to: {save_path}")

        elif cfg.processing_mode == "apply_drowsiness":
            applier = DrowsinessLabelApplier(
                cfg.training_folder,
                cfg.output_folder / "Training")
            applier.apply_labels()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)