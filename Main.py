from Workspace import *
import time
if __name__ == "__main__":

    def second_task():
        while True:
            fps = Utils.calculate_fps()

            ret, frame = camera.read()
            processed_frame, face_mesh_coords = image_processor.process_face_image(frame)

            perclos, ear = find_perclos.find_parameter(face_mesh_coords)
            is_jawning, yawn_counter, mar = find_yawn.find_parameter(face_mesh_coords)
            roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)
            saccade_velocity = find_saccade_velocity.find_parameter(face_mesh_coords)

            # Utils.render_face_coordinates(coordinates_parser, face_plotter, face_mesh_coords)

            cv2.imshow('Drowsiness detection', processed_frame)

            packet = {
                "MAR": perclos,
                "Yawning": is_jawning,
                "Roll": roll,
                "Pitch": pitch,
                "EAR": ear,
                "PERCLOS": perclos
            }

            gui_display.update_parameters(mar, is_jawning, roll, pitch, ear, perclos, yawn_counter, fps)

            sql_saver.save_to_csv(packet)

            # Nacisnij 'q', aby wyjsc
            if cv2.waitKey(1) & 0xFF == ord('q'):
                gui.running = False
                break

    pathlib.PosixPath = Utils.fix_pathlib()

    # Załadowanie klasy ImageProcessor do przetwarzania obrazu
    image_processor = ImageProcessor()
    crop_size = (224,224)
    text_color = [240, 10, 10]
    text_parameters = (cv2.FONT_HERSHEY_DUPLEX,1,text_color,2)

    # model_loader = ModelLoader()
    # models = model_loader.load_models()

    coordinates_parser = CoordinatesParser()
    # face_plotter = face_plotter.FacePlotter() #TODO Poprawić to, aby też się wyświetlało w GUI
    sql_saver = SqlSaver()
    os.system('cls')

    # Initialise camera for video capture
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
    # saccs = np.zeros(1)

    gui_display = GUI()
    second_threat = threading.Thread(target=second_task, daemon=True)
    second_threat.start()

    gui_display.start()

    # camera.release()
    # cv2.destroyAllWindows()



