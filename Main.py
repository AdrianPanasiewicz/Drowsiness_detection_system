from Workspace import *
import threading

if __name__ == "__main__":

    def analyze_face_parameters():
        while True:
            fps = Utils.calculate_fps()

            ret, frame = camera.read()
            processed_frame, face_mesh_coords = image_processor.process_face_image(frame)

            perclos, ear = find_perclos.find_parameter(face_mesh_coords)
            is_jawning, yawn_counter, mar = find_yawn.find_parameter(face_mesh_coords)
            roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)

            face_plotter_inst = gui_display.get_face_plotter()
            Utils.render_face_coordinates(coordinates_parser, face_plotter_inst, face_mesh_coords)
            gui_display.set_face_plotter(face_plotter_inst)

            packet = {
                "MAR": perclos,
                "Yawning": is_jawning,
                "Roll": roll,
                "Pitch": pitch,
                "EAR": ear,
                "PERCLOS": perclos
            }

            gui_display.queue_parameters(mar, is_jawning, roll, pitch, ear, perclos, yawn_counter, fps)
            gui_display.queue_image(processed_frame)

            sql_saver.save_to_csv(packet)

            if 0xFF == ord('q'):
                gui.running = False
                break

    pathlib.PosixPath = Utils.fix_pathlib()
    image_processor = ImageProcessor()

    # model_loader = ModelLoader()
    # models = model_loader.load_models()

    coordinates_parser = CoordinatesParser()
    sql_saver = SqlSaver()
    os.system('cls')

    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError

    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    perclos_threshold = 0.2
    yawn_threshold = 0.55

    find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
    find_yawn = yawn_finder.YawnFinder(yawn_threshold)
    find_face_tilt = face_angle_finder.FaceAngleFinder()
    find_saccade_velocity = saccade_speed_velocity.SaccadeVelocityFinder()

    gui_display = GUI()
    second_threat = threading.Thread(target=analyze_face_parameters, daemon=True)
    second_threat.start()

    gui_display.start()



