from Workspace import *
if __name__ == "__main__":

    pathlib.PosixPath = Utils.fix_pathlib()

    # Załadowanie klasy ImageProcessor do przetwarzania obrazu
    image_processor = ImageProcessor()
    crop_size = (224,224)
    text_color = [240, 10, 10]
    text_parameters = (cv2.FONT_HERSHEY_DUPLEX,1,text_color,2)

    # model_loader = ModelLoader()
    # models = model_loader.load_models()

    coordinates_parser = CoordinatesParser()
    face_plotter = face_plotter.FacePlotter()
    os.system('cls')

    # Initialise camera for video capture
    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError

    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    perclos_threshold = 0.4
    yawn_threshold = 0.6

    find_perclos = perclos_finder.PerclosFinder(perclos_threshold)
    find_jawn = jawn_finder.JawnFinder(yawn_threshold)
    find_face_tilt = face_angle_finder.FaceAngleFinder()
    find_saccade_velocity = saccade_speed_velocity.SaccadeVelocityFinder()
    # saccs = np.zeros(1)

    while True:
        fps = Utils.calculate_fps()

        ret, frame = camera.read()
        processed_frame, face_mesh_coords = image_processor.process_face_image(frame)

        perclos = find_perclos.find_parameter(face_mesh_coords)
        is_jawning, jawn_counter = find_jawn.find_parameter(face_mesh_coords)
        roll, pitch = find_face_tilt.find_parameter(face_mesh_coords)
        saccade_velocity = find_saccade_velocity.find_parameter(face_mesh_coords)

        # os.system('cls')
        # print(f"PERCLOS =\t{round(perclos,2)}")

        Utils.render_face_coordinates(coordinates_parser, face_plotter, face_mesh_coords)

        cv2.putText(processed_frame, f"Saccade speed: {round(saccade_velocity, 3)}", (15, 60), *text_parameters)
        cv2.putText(processed_frame, f"PERCLOS: {int(perclos*100)}%", (15, 90), *text_parameters)
        cv2.putText(processed_frame, f"Jawn: {is_jawning}", (15, 120), *text_parameters)
        cv2.putText(processed_frame, f"Jawn counter: {jawn_counter}", (15, 150), *text_parameters)
        cv2.putText(processed_frame, f"Roll: {round(roll, 2)}", (15, 180), *text_parameters)
        cv2.putText(processed_frame, f"Pitch: {round(pitch, 2)}", (15, 210), *text_parameters)
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (15, 240), *text_parameters)
        cv2.imshow('Drowsiness detection', processed_frame)

        # if len(saccs) >= 20:
        #     saccs = np.pitch(saccs, -1)
        #     saccs[-1] = saccade_velocity
        # else:
        #     saccs = np.append(saccs, saccade_velocity)
        #
        # plt.axis([0, 20, -30, 0])
        # plt.plot(saccs)
        # plt.pause(0.001)
        # plt.clf()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



