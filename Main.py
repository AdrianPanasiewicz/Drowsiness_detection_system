from Workspace import *
import threading

if __name__ == "__main__":

    def analyze_face_parameters():
        while True:
            # Obliczanie klatek na sekundę (FPS)
            fps = Utils.calculate_fps()

            # Pobranie klatki z kamery i przetworzenie obrazu (m.in. wykrycie siatki twarzy)
            ret, frame = camera.read()
            processed_frame, face_mesh_coords = image_processor.process_face_image(frame)

            # Wyznaczenie poszczególnych wskaźników (PERCLOS, EAR, ziewanie, pochylenie głowy)
            perclos, ear = find_perclos.find_parameter(face_mesh_coords)
            is_jawning, yawn_counter, mar = find_yawn.find_parameter(face_mesh_coords)
            roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)

            # Aktualizacja wizualizacji 3D i parametrów w GUI
            face_plotter_inst = gui_display.get_face_plotter()
            Utils.render_face_coordinates(coordinates_parser, face_plotter_inst, face_mesh_coords)
            gui_display.set_face_plotter(face_plotter_inst)
            gui_display.queue_parameters(mar, is_jawning, roll, pitch, ear, perclos, yawn_counter, fps)
            gui_display.queue_image(processed_frame)

            # Zapis wyników do pliku CSV
            packet = {"MAR": perclos, "Yawning": is_jawning, "Roll": roll, "Pitch": pitch, "EAR": ear, "PERCLOS": perclos}
            sql_saver.save_to_csv(packet)

            # Wyjście z pętli po wciśnięciu 'q'
            if 0xFF == ord('q'):
                gui.running = False
                break

    # Poprawka typów ścieżek w Windows
    pathlib.PosixPath = Utils.fix_pathlib()

    # Inicjalizacja klas związanych z systemem
    image_processor = ImageProcessor()
    coordinates_parser = CoordinatesParser()
    sql_saver = SqlSaver()
    os.system('cls')

    # Inicjalizacja kamery
    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError
    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    # Ustawienia progów dla PERCLOS i ziewania
    perclos_threshold = 0.2
    yawn_threshold = 0.55

    # Obiekty do wyliczania PERCLOS, ziewania, pochylenia głowy, itp.
    find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
    find_yawn = yawn_finder.YawnFinder(yawn_threshold)
    find_face_tilt = face_angle_finder.FaceAngleFinder()
    find_saccade_velocity = saccade_speed_velocity.SaccadeVelocityFinder()

    # Utworzenie GUI i uruchomienie wątku do analizy
    gui_display = GUI()
    second_thread = threading.Thread(target=analyze_face_parameters, daemon=True)
    second_thread.start()

    # Start głównej pętli zdarzeń w GUI
    gui_display.start()
