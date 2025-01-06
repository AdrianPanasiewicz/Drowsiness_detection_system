from Workspace import *
import threading
import os
import sys
import pathlib
import cv2
import pandas as pd

def camera_mode(camera, image_processor_inst, coordinates_parser_inst, sql_saver_inst, find_perclos,
                find_yawn, find_face_tilt, random_forest_classifier, gui_display_inst=None):
    while True:
        # Oblicza liczbę klatek na sekundę (FPS)
        fps = Utils.calculate_fps()

        # Przechwytuje klatkę z kamery i przetwarza obraz
        ret, frame = camera.read()
        if not ret:
            print("Nie udało się odczytać klatki z kamery.")
            break

        processed_frame, face_mesh_coords = image_processor_inst.process_face_image(frame)

        # Oblicza wskaźniki senności
        perclos, ear = find_perclos.find_parameter(face_mesh_coords)
        is_jawning, yawn_counter, mar = find_yawn.find_parameter(face_mesh_coords)
        roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)

        if face_mesh_coords.multi_face_landmarks:
            cols = ["MAR", "EAR", "Roll", "Pitch"]
            data_for_prediction = pd.DataFrame([[mar, ear, roll, pitch]], columns=cols)
            prediction = random_forest_classifier.moving_mode_value_prediction(data_for_prediction)
        else:
            prediction = None

        # Aktualizuje GUI za pomocą przetworzonych danych
        if gui_display_inst:
            face_plotter_inst = gui_display_inst.get_face_plotter()
            Utils.render_face_coordinates(coordinates_parser_inst, face_plotter_inst, face_mesh_coords)
            gui_display_inst.set_face_plotter(face_plotter_inst)
            gui_display_inst.queue_parameters(prediction, mar, is_jawning, roll, pitch, ear, perclos, yawn_counter, fps)
            gui_display_inst.queue_image(processed_frame)

        # Zapisuje wyniki do pliku CSV
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


def image_mode(image_folder, image_processor_inst, sql_saver_inst, perclos_finder_inst,
               yawn_finder_inst, face_angle_finder_inst, random_forest_classifier):

    image_paths = list(
        pathlib.Path(image_folder).glob('*.*'))

    map_drowsiness_label = {"0":"Not_drowsy", "1":"Drowsy", "2":"Drowsy"}

    for index, image_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Nie udało się odczytać obrazu: {image_path}")
            continue

        processed_frame, face_mesh_coords = image_processor_inst.process_face_image(frame)

        if face_mesh_coords:

            # Oblicza wskaźniki senności
            perclos, ear = perclos_finder_inst.find_parameter(face_mesh_coords)
            is_jawning, yawn_counter, mar = yawn_finder_inst.find_parameter(face_mesh_coords)
            roll, pitch = face_angle_finder_inst.find_parameter(face_mesh_coords)

            if 0.1 <= ear < 0.2:
                cols = ["MAR", "EAR", "Roll", "Pitch"]
                data_for_prediction = pd.DataFrame([[mar, ear, roll, pitch]], columns=cols)
                prediction = random_forest_classifier.moving_mode_value_prediction(data_for_prediction)
            elif ear >= 0.2:
                prediction = True
            else:
                prediction = False

            # Do przetwarzania bazy danych do trenowania modelu
            image_name = image_path.name
            label_path = image_path.parent.parent / 'labels' / f'{image_name[:-4]}.txt'
            with open(label_path, 'r') as label_file:
                drowsiness_label = map_drowsiness_label[label_file.read(1)]

            # Zapisuje wyniki do pliku CSV, w tym nazwę pliku obrazu
            packet = {
                "Image": str(image_path.name),
                "MAR": mar,
                "Yawning": is_jawning,
                "Roll": roll,
                "Pitch": pitch,
                "EAR": ear,
                "Drowsy": prediction
            }
            sql_saver_inst.save_to_csv(packet)

        os.system('cls' if os.name == 'nt' else 'clear')
        if index % 100 == 0:
            print(f"Przetworzono obrazów: {index}/{len(image_paths)}")

    print(f"Przetwarzanie zakończone. Wyniki zapisane w {sql_saver_inst.saving_path}")


def main():

    mode = "camera"
    results_name = "results.csv"

    image_folder = pathlib.Path(r"C:\Users\adria\Documents\drowsiness_dataset")
    train_folder = image_folder / r"train\images"
    val_folder = image_folder / r"valid\images"
    test_folder = image_folder / "test\images"
    training_name = "training_data.csv"
    validating_name = "validating_data.csv"
    testing_name = "testing_data.csv"


    # Naprawia pathlib dla systemu Windows, jeśli to konieczne
    pathlib.PosixPath = Utils.fix_pathlib()

    # Inicjalizuje klasy związane z systemem
    image_processor_inst = ImageProcessor()
    coordinates_parser_inst = CoordinatesParser()

    os.system('cls' if os.name == 'nt' else 'clear')

    # Ustawia progi dla PERCLOS i ziewania
    perclos_threshold = 0.2
    yawn_threshold = 0.5

    if mode == 'camera':
        sql_saver = SqlSaver(results_name)

        find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
        find_yawn = yawn_finder.YawnFinder(yawn_threshold)
        find_face_tilt = face_angle_finder.FaceAngleFinder()
        random_forest_classifier = RandomForest(activation_certainty=0.5, prediction_memory_size=50)

        # Inicjalizuje kamerę
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje')
        except TypeError as e:
            print(e)
            sys.exit(1)

        # Inicjalizuje GUI
        gui_display = GUI()

        # Rozpoczyna analizę w osobnym wątku
        analysis_thread = threading.Thread(
            target=camera_mode,
            args=(camera, image_processor_inst, coordinates_parser_inst, sql_saver, find_perclos, find_yawn,
                  find_face_tilt, random_forest_classifier, gui_display),
            daemon=True
        )
        analysis_thread.start()

        # Rozpoczyna pętlę zdarzeń GUI
        gui_display.start()

        # Zwolnienie kamery po zakończeniu
        camera.release()

    elif mode == 'image':
        sql_saver_training_inst = SqlSaver(training_name)
        sql_saver_validating_inst = SqlSaver(validating_name)
        sql_saver_testing_inst = SqlSaver(testing_name)
        random_forest_classifier = RandomForest(activation_certainty=0.5, prediction_memory_size=50)

        find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
        find_yawn = yawn_finder.YawnFinder(yawn_threshold, is_image_mode=True)
        find_face_tilt = face_angle_finder.FaceAngleFinder()

        if not pathlib.Path(image_folder).is_dir():
            print(f"Podany folder z obrazami nie istnieje lub nie jest katalogiem: {image_folder}")
            sys.exit(1)

        # Przetwarza obrazy
        image_mode(
            train_folder,
            image_processor_inst,
            sql_saver_training_inst,
            find_perclos,
            find_yawn,
            find_face_tilt,
            random_forest_classifier
        )

        image_mode(
            val_folder,
            image_processor_inst,
            sql_saver_validating_inst,
            find_perclos,
            find_yawn,
            find_face_tilt,
            random_forest_classifier
        )

        image_mode(
            test_folder,
            image_processor_inst,
            sql_saver_testing_inst,
            find_perclos,
            find_yawn,
            find_face_tilt,
            random_forest_classifier
        )

if __name__ == "__main__":
    main()
