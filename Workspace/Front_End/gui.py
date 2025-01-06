import customtkinter
import queue
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from Workspace.Front_End.face_plotter import FacePlotter
from typing import Optional, Union


class GUI:
    """
    Klasa GUI zarządza głównym oknem aplikacji, umożliwia wyświetlanie
    podglądu z kamery, wizualizację 3D twarzy oraz prezentację obliczonych
    parametrów (m.in. MAR, EAR, PERCLOS). Zapewnia także funkcjonalność
    zmiany stylu interfejsu i wyświetlanie informacji o programie.
    """

    def __init__(self) -> None:
        """
        Inicjalizuje główne okno aplikacji i wszystkie jego komponenty,
        w tym ramki (frames), etykiety (labels) i płótna (canvas).
        Ustawia także kolejki (Queue) do obsługi asynchronicznego
        wyświetlania obrazu z kamery oraz zaktualizowanych parametrów.
        """
        # Flaga kontrolująca działanie głównej pętli programu.
        self.running = True

        # Kolejki do przechowywania danych o parametrach i obrazu z kamery.
        self.data_queue = queue.Queue()
        self.image_queue = queue.Queue()

        # Bieżący obraz z kamery, przetworzony dla tkinter.
        self.current_image = None

        # Ustawienie rozmiarów głównego okna.
        self.width = 1450
        self.height = 650

        # Konfiguracja wyglądu interfejsu (ciemny/jasny/systemowy).
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        # Tworzenie głównego okna aplikacji.
        self.window = customtkinter.CTk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("System wykrywania senności")
        self.window.resizable(False, False)

        # Ścieżka do bieżącego pliku – używana np. do ładowania zasobów (obrazów tła).
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # Ładowanie i ustawianie obrazu tła w głównym oknie (użycie metody place).
        self.bg_image = customtkinter.CTkImage(
            Image.open(self.current_path + "/Background_Images/gray_bg_img.jpeg"),
            size=(self.width, self.height)
        )
        self.bg_image_label = customtkinter.CTkLabel(self.window, text="", image=self.bg_image)
        self.bg_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Dolny panel z listą wyboru trybu (ciemny/jasny/systemowy) i przyciskiem informacyjnym.
        self.bottom_frame = customtkinter.CTkFrame(self.window)
        self.bottom_frame.pack(side="bottom", fill="x")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            self.bottom_frame,
            values=["Ciemny", "Jasny", "Systemowy"],
            command=self.change_appearance
        )
        self.appearance_mode_menu.pack(side="left", padx=20, pady=10)

        self.info_button = customtkinter.CTkButton(
            self.bottom_frame,
            text="O programie",
            command=self.show_info_window
        )
        self.info_button.pack(side="right", padx=20, pady=10)

        # Ramka (frame) z lewej strony do wyświetlania obrazu z kamery.
        self.webcam_frame = customtkinter.CTkFrame(self.window)
        self.webcam_frame.pack(pady=20, padx=20, fill="both", expand=True, side="left")

        # Etykieta tytułowa w ramce kamery.
        self.webcam_label = customtkinter.CTkLabel(
            self.webcam_frame,
            text="Kamera",
            font=("JetBrains Mono", 40, 'bold')
        )
        self.webcam_label.pack(pady=20, padx=20)

        # Obszar (Canvas) do wyświetlania bieżącego obrazu z kamery.
        self.webcam_canvas = customtkinter.CTkCanvas(self.webcam_frame, width=640, height=480)
        self.webcam_canvas.pack(pady=25, padx=5, anchor="center")

        # Ramka (frame) po prawej stronie do osadzenia wizualizacji 3D twarzy.
        self.plot_frame = customtkinter.CTkFrame(self.window)
        self.plot_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")

        # Etykieta tytułowa dla wizualizacji 3D.
        self.plot_label = customtkinter.CTkLabel(
            self.plot_frame,
            text="Wizualizacja 3D twarzy",
            font=("JetBrains Mono", 40, 'bold')
        )
        self.plot_label.pack(pady=20, padx=20)

        # Inicjalizacja obiektu Figure (matplotlib) i dodanie osi 3D.
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Umieszczenie wykresu Matplotlib w interfejsie TKinter przy użyciu FigureCanvasTkAgg.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Utworzenie instancji FacePlotter do wyświetlania twarzy 3D i uruchomienie animacji.
        self.face_plotter_inst = FacePlotter(
            figure=self.fig,
            axes3d=self.ax,
            canvas=self.canvas,
            root=self.window
        )
        self.face_plotter_inst.start_animation(interval=120)

        # Ramka (frame) do wyświetlania parametrów obliczanych na podstawie obrazu twarzy.
        self.params_frame = customtkinter.CTkFrame(self.window)
        self.params_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")

        # Górna część ramki parametrów.
        self.params_info_upper_layer = customtkinter.CTkFrame(self.params_frame)
        self.params_info_upper_layer.pack(pady=20, padx=20, fill="both", expand=True, side="top")

        # Środkowa część ramki parametrów, w której umieszczane są etykiety i wartości.
        self.params_info_mid_layer = customtkinter.CTkFrame(self.params_frame)
        self.params_info_mid_layer.pack(pady=10, padx=20, fill="both", expand=True)

        # Etykieta tytułowa w ramce parametrów.
        self.params_title = customtkinter.CTkLabel(
            self.params_info_upper_layer,
            text="Parametry:",
            font=("JetBrains Mono", 40, 'bold')
        )
        self.params_title.pack(pady=20, padx=20)

        # Funkcja pomocnicza do tworzenia etykiet w panelu parametrów.
        def _create_label_in_params(text: str, width: int = 150) -> customtkinter.CTkLabel:
            label = customtkinter.CTkLabel(
                self.params_info_mid_layer,
                text=text,
                width=width,
                font=("JetBrains Mono", 20),
                anchor="w"
            )
            return label

        # Listy przechowujące etykiety (nazwa parametru) i ich wartości.
        self.label_list = []
        self.value_list = []
        parameters_name = ["MAR:", "Ziewanie:", "Liczba ziewnięć:", "Roll:", "Pitch:", "EAR:", "PERCLOS:"]

        # Tworzenie etykiet i przypisywanie ich do gridu.
        for i in range(len(parameters_name)):
            self.label_list.append(_create_label_in_params(parameters_name[i]))
            self.value_list.append(_create_label_in_params("None"))
            self.label_list[i].grid(row=i, column=0, sticky='W', padx=(25, 0), pady=3)
            self.value_list[i].grid(row=i, column=1, sticky='W', padx=(25, 10), pady=3)

        # Ramka (frame) do prezentowania stanu operatora.
        self.prediction_frame = customtkinter.CTkFrame(self.params_frame)
        self.prediction_frame.pack(pady=20, padx=20, fill="both", expand=True, side="bottom")

        # Etykieta tytułowa dotycząca stanu operatora.
        self.prediction_title = customtkinter.CTkLabel(
            self.prediction_frame,
            text=f"Stan operatora:",
            font=("JetBrains Mono", 30, 'bold'),
        )
        self.prediction_title.pack(pady=10, padx=20)

        # Etykieta wyświetlająca aktualny stan (np. "Senny", "Czujny", "Brak operatora").
        self.prediction_info = customtkinter.CTkLabel(
            self.prediction_frame,
            text=f"None",
            font=("JetBrains Mono", 30, 'bold'),
        )
        self.prediction_info.pack(pady=10, padx=20)

        # Wywołanie metod cyklicznych, odpowiedzialnych za aktualizację obrazu oraz etykiet.
        self.update_webcam()
        self.update_labels()

    def show_info_window(self) -> None:
        """
        Wyświetla okno informacyjne „O programie”, zawierające dane na temat autora,
        celu aplikacji i sposobu jej obsługi.
        """
        info_window = customtkinter.CTkToplevel(self.window)
        info_window.title("O programie")
        info_window.geometry("500x380")

        # Ustawienie parametru topmost, aby okno z informacjami było na wierzchu.
        info_window.attributes("-topmost", 1)

        info_test = """        Autor: Adrian Paweł Panasiewicz  
        Tytuł pracy dyplomowej: Projekt wstępny systemu bezpieczeństwa do wykrywania 
                                                senności u pilotów bezzałogowych 
                                                statków powietrznych  

        Cel aplikacji:
        Aplikacja została zaprojektowana w celu monitorowania stanu senności 
        operatorów dronów w czasie rzeczywistym. Analizuje obraz z kamery 
        i wyświetla parametry takie jak PERCLOS, ziewanie, czy pochylenie 
        głowy. W przypadku wykrycia krytycznych wartości system generuje 
        alerty dźwiękowe i wizualne.

        Jak obsługiwać aplikację:
        1. Uruchom plik wykonywalny aplikacji na systemie Windows 10/11.
        2. Podłącz kamerę zgodną z minimalnymi wymaganiami (720p, 30 FPS).
        3. Ustaw kamerę tak, aby rejestrowała twarz operatora w dobrych 
           warunkach oświetleniowych.
        4. Obserwuj dane wyjściowe na interfejsie graficznym aplikacji (GUI).

        Uwaga:
        - Aplikacja obsługuje tylko jedną twarz w kadrze.
        - Stabilne oświetlenie i minimalne ruchy kamery poprawiają dokładność analizy.
        - Wszelkie dane są zapisywane w bazie danych, umożliwiając późniejszą analizę.
        """

        info_label = customtkinter.CTkLabel(
            info_window,
            text=info_test,
            justify="left",
            font=("JetBrains Mono", 12)
        )
        info_label.pack(pady=20, padx=20)

    def queue_image(self, image: 'np.ndarray') -> None:
        """
        Dodaje nowy obraz (np. z kamery) do kolejki image_queue, skąd
        będzie pobierany i wyświetlany w metodzie update_webcam().

        :param image: Obraz w formacie kompatybilnym z OpenCV (np. BGR).
        :type image: numpy.ndarray
        """
        self.image_queue.put(image)

    def get_face_plotter(self) -> FacePlotter:
        """
        Zwraca instancję klasy FacePlotter wykorzystywaną do wizualizacji 3D twarzy.

        :return: Instancja FacePlotter.
        :rtype: FacePlotter
        """
        return self.face_plotter_inst

    def set_face_plotter(self, face_plotter_inst: FacePlotter) -> None:
        """
        Umożliwia wymianę lub ponowne ustawienie instancji FacePlotter.

        :param face_plotter_inst: Nowa instancja FacePlotter.
        :type face_plotter_inst: FacePlotter
        """
        self.face_plotter_inst = face_plotter_inst

    def queue_parameters(
        self,
        prediction: Optional[bool],
        mar: float,
        is_yawning: bool,
        roll: float,
        pitch: float,
        ear: float,
        perclos: float,
        yawn_counter: Union[int, float],
        fps: Union[int, float]
    ) -> None:
        """
        Dodaje zestaw parametrów (m.in. MAR, EAR, PERCLOS, FPS) do kolejki data_queue,
        skąd będą pobrane i wyświetlane w metodzie update_labels().

        :param prediction: Informacja, czy operator jest senny (True), brak operatora (None) lub czujny (False).
        :type prediction: bool lub None
        :param mar: Mouth Aspect Ratio (MAR).
        :type mar: float
        :param is_yawning: Informacja, czy aktualnie wykryto ziewanie.
        :type is_yawning: bool
        :param roll: Kąt przechyłu głowy (roll).
        :type roll: float
        :param pitch: Kąt pochylenia głowy (pitch).
        :type pitch: float
        :param ear: Eye Aspect Ratio (EAR).
        :type ear: float
        :param perclos: Wskaźnik PERCLOS (procent czasu, gdy oczy są zamknięte).
        :type perclos: float
        :param yawn_counter: Licznik ziewnięć w obecnej sesji.
        :type yawn_counter: int lub float
        :param fps: Liczba klatek na sekundę (Frames Per Second).
        :type fps: float lub int
        """
        self.data_queue.put(
            (prediction, mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps)
        )

    def change_appearance(self, new_appearance_mode: str) -> None:
        """
        Zmienia styl (jasny/ciemny/systemowy) interfejsu za pomocą funkcji
        z biblioteki customtkinter. Jednocześnie aktualizuje obraz tła w zależności
        od wybranego stylu.

        :param new_appearance_mode: Wybrany styl: "Jasny", "Ciemny" lub "Systemowy".
        :type new_appearance_mode: str
        """
        mode_dict = {"Jasny": "Light", "Ciemny": "Dark", "Systemowy": "System"}
        customtkinter.set_appearance_mode(mode_dict[new_appearance_mode])

        if mode_dict[new_appearance_mode] == "Light":
            bg_image_path = self.current_path + "/Background_Images/white_bg_img.png"
        elif mode_dict[new_appearance_mode] == "Dark":
            bg_image_path = self.current_path + "/Background_Images/gray_bg_img.jpeg"
        else:
            bg_image_path = self.current_path + "/Background_Images/gray_bg_img.jpeg"

        # Aktualizacja obrazka tła (ponowne jego wczytanie i przypisanie do etykiety).
        self.bg_image = customtkinter.CTkImage(
            Image.open(bg_image_path),
            size=(self.width, self.height)
        )
        self.bg_image_label.configure(image=self.bg_image)

    def update_webcam(self) -> None:
        """
        Cykl aktualizacji obrazu w oknie 'Kamera'. Metoda co około 33 ms (30 FPS)
        pobiera z kolejki najnowszy obraz, konwertuje go do formatu RGB i wyświetla
        na płótnie (Canvas). Następnie ponownie planuje wywołanie samej siebie
        (window.after()).
        """
        try:
            while not self.image_queue.empty():
                image = self.image_queue.get_nowait()
                # Konwersja obrazu z BGR (OpenCV) do RGB, a następnie do formatu PIL.
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.current_image = ImageTk.PhotoImage(image)
                self.webcam_canvas.create_image(0, 0, image=self.current_image, anchor="nw")

        except queue.Empty:
            pass

        if self.running:
            # Zaplanowanie ponownego wywołania metody po 33 ms (≈30 FPS).
            self.window.after(33, self.update_webcam)

    def update_labels(self) -> None:
        """
        Cykl aktualizacji etykiet z parametrami w panelu 'Parametry'.
        Metoda pobiera z kolejki data_queue najnowszy zestaw obliczonych
        parametrów i wyświetla je w odpowiednich etykietach. Następnie
        ponownie planuje wywołanie samej siebie (window.after()).
        """
        try:
            while not self.data_queue.empty():
                (
                    prediction,
                    mar,
                    is_yawning,
                    roll,
                    pitch,
                    ear,
                    perclos,
                    yawn_counter,
                    fps
                ) = self.data_queue.get_nowait()

                # Uaktualnienie wartości MAR (Mouth Aspect Ratio).
                self.value_list[0].configure(
                    text=f"{round(mar, 2)}"
                )

                # Informacja o ziewaniu (Obecne/Brak).
                self.value_list[1].configure(
                    text=f'{"Obecne" if is_yawning else "Brak"}'
                )

                # Liczba ziewnięć zarejestrowanych podczas sesji.
                self.value_list[2].configure(
                    text=f'{yawn_counter}'
                )

                # Kąt przechyłu głowy (roll).
                self.value_list[3].configure(
                    text=f"{round(roll, 2)}"
                )

                # Kąt pochylenia głowy (pitch).
                self.value_list[4].configure(
                    text=f"{round(pitch, 2)}"
                )

                # Wskaźnik EAR (Eye Aspect Ratio).
                self.value_list[5].configure(
                    text=f"{round(ear, 2)}"
                )

                # Wskaźnik PERCLOS wyrażony w procentach.
                self.value_list[6].configure(
                    text=f"{round(100 * perclos, 2)}%"
                )

                # Interpretacja stanu operatora (Senny/Czujny/Brak operatora).
                if prediction:
                    pred_text = "Senny"
                    text_color = 'red'
                elif prediction is None:
                    pred_text = "Brak operatora"
                    text_color = 'yellow'
                else:
                    pred_text = "Czujny"
                    text_color = '#0AAA08'

                self.prediction_info.configure(
                    text=pred_text,
                    text_color=text_color
                )
        except queue.Empty:
            pass

        if self.running:
            # Zaplanowanie ponownego wywołania metody po 200 ms.
            self.window.after(200, self.update_labels)

    def start(self) -> None:
        """
        Rozpoczyna główną pętlę zdarzeń Tkinter, dzięki czemu interfejs użytkownika
        pozostaje aktywny i reaguje na działania (np. odświeżanie obrazu czy klikanie przycisków).
        """
        self.window.mainloop()
