from Workspace import *
import threading
import os
import sys
import pathlib
import cv2

def camera_mode(camera, image_processor_inst, coordinates_parser_inst, sql_saver_inst, find_perclos,
                find_yawn, find_face_tilt, gui_display_inst=None):
    while True:
        # Calculate Frames Per Second (FPS)
        fps = Utils.calculate_fps()

        # Capture frame from the camera and process the image
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        processed_frame, face_mesh_coords = image_processor_inst.process_face_image(frame)

        # Compute drowsiness indicators
        perclos, ear = find_perclos.find_parameter(face_mesh_coords)
        is_jawning, yawn_counter, mar = find_yawn.find_parameter(face_mesh_coords)
        roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)

        # Update GUI with processed data
        if gui_display_inst:
            face_plotter_inst = gui_display_inst.get_face_plotter()
            Utils.render_face_coordinates(coordinates_parser_inst, face_plotter_inst, face_mesh_coords)
            gui_display_inst.set_face_plotter(face_plotter_inst)
            gui_display_inst.queue_parameters(mar, is_jawning, roll, pitch, ear, perclos, yawn_counter, fps)
            gui_display_inst.queue_image(processed_frame)

        # Save results to CSV
        packet = {
            "MAR": mar,
            "Yawning": is_jawning,
            "YawnCounter": yawn_counter,
            "Roll": roll,
            "Pitch": pitch,
            "EAR": ear,
            "PERCLOS": perclos,
        }
        sql_saver_inst.save_to_csv(packet)


def process_images_mode(image_folder, image_processor_inst, sql_saver_inst, perclos_finder_inst,
                        yawn_finder_inst, face_angle_finder_inst):
    image_paths = list(
        pathlib.Path(image_folder).glob('*.*'))  # Adjust the pattern as needed (e.g., '*.jpg', '*.png')
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Failed to read image: {image_path}")
            continue

        processed_frame, face_mesh_coords = image_processor_inst.process_face_image(frame)

        # Compute drowsiness indicators
        perclos, ear = perclos_finder_inst.find_parameter(face_mesh_coords)
        is_jawning, yawn_counter, mar = yawn_finder_inst.find_parameter(face_mesh_coords)
        roll, pitch = face_angle_finder_inst.find_parameter(face_mesh_coords)

        # Save results to CSV, including the image filename
        packet = {
            "Image": str(image_path.name),
            "MAR": mar,
            "Yawning": is_jawning,
            "Roll": roll,
            "Pitch": pitch,
            "EAR": ear,
        }
        sql_saver_inst.save_to_csv(packet)

    print(f"Processing completed. Results saved to {sql_saver_inst.saving_path}")

def main():

    mode = "camera"
    image_folder = ""

    # Fix pathlib for Windows if necessary
    pathlib.PosixPath = Utils.fix_pathlib()

    # Initialize system-related classes
    image_processor_inst = ImageProcessor()
    coordinates_parser_inst = CoordinatesParser()
    sql_saver_inst = SqlSaver()
    os.system('cls' if os.name == 'nt' else 'clear')

    # Threshold settings for PERCLOS and yawning
    perclos_threshold = 0.2
    yawn_threshold = 0.55

    # Initialize parameter finders
    find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
    find_yawn = yawn_finder.YawnFinder(yawn_threshold)
    find_face_tilt = face_angle_finder.FaceAngleFinder()

    if mode == 'camera':
        # Initialize camera
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje')
        except TypeError as e:
            print(e)
            sys.exit(1)

        # Initialize GUI
        gui_display = GUI()

        # Start analysis in a separate thread
        analysis_thread = threading.Thread(
            target=camera_mode,
            args=(camera, image_processor_inst, coordinates_parser_inst, sql_saver_inst, find_perclos, find_yawn,
                  find_face_tilt, gui_display),
            daemon=True
        )
        analysis_thread.start()

        # Start the GUI event loop
        gui_display.start()

        # Release the camera upon exit
        camera.release()

    elif mode == 'image':

        image_folder = image_folder
        if not pathlib.Path(image_folder).is_dir():
            print(f"The provided image folder does not exist or is not a directory: {image_folder}")
            sys.exit(1)

        # Process images
        process_images_mode(
            image_folder,
            image_processor_inst,
            sql_saver_inst,
            find_perclos,
            find_yawn,
            find_face_tilt
        )

if __name__ == "__main__":
    main()

