import customtkinter
import queue
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import cv2
from Workspace.Front_End.face_plotter import FacePlotter

class GUI:
    def __init__(self):
        # Flaga określająca, czy główna pętla programu powinna się uruchamiać.
        self.running = True

        # Kolejki, w których będą przechowywane dane parametry i klatki obrazu do wyświetlenia w GUI.
        self.data_queue = queue.Queue()
        self.image_queue = queue.Queue()

        # Przechowywana aktualna klatka z kamery (w formacie zgodnym z tkinter).
        self.current_image = None

        # Ustawienie rozmiarów głównego okna.
        self.width = 1450
        self.height = 650

        # Konfiguracja wyglądu (tryb ciemny/jasny/systemowy) za pomocą biblioteki customtkinter.
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        # Tworzenie głównego okna aplikacji.
        self.window = customtkinter.CTk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("System wykrywania senności")
        self.window.resizable(False, False)

        # Ścieżka do bieżącego pliku (wykorzystywana np. do ładowania obrazów tła).
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # Ładowanie obrazu tła dla głównego okna i umieszczenie go jako tło za pomocą `place`.
        self.bg_image = customtkinter.CTkImage(
            Image.open(self.current_path + "/gray_bg_img.jpeg"),
            size=(self.width, self.height)
        )
        self.bg_image_label = customtkinter.CTkLabel(self.window, text="", image=self.bg_image)
        self.bg_image_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Dolny panel (frame) z menu wyboru trybu wyglądu i przyciskiem informacji o programie.
        self.bottom_frame = customtkinter.CTkFrame(self.window)
        self.bottom_frame.pack(side="bottom", fill="x")

        # Menu wyboru trybu wyglądu (Ciemny/Jasny/Systemowy).
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            self.bottom_frame,
            values=["Ciemny", "Jasny", "Systemowy"],
            command=self.change_appearance
        )
        self.appearance_mode_menu.pack(side="left", padx=20, pady=10)

        # Przycisk wyświetlający okno „O programie”.
        self.info_button = customtkinter.CTkButton(
            self.bottom_frame,
            text="O programie",
            command=self.show_info_window
        )
        self.info_button.pack(side="right", padx=20, pady=10)

        # Sekcja po lewej stronie okna – ramka (frame) wyświetlająca obraz z kamery.
        self.webcam_frame = customtkinter.CTkFrame(self.window)
        self.webcam_frame.pack(pady=20, padx=20, fill="both", expand=True, side="left")

        # Etykieta – nagłówek informujący, że poniżej widać „Kamera”.
        self.webcam_label = customtkinter.CTkLabel(
            self.webcam_frame, text="Kamera", font=("JetBrains Mono", 40, 'bold')
        )
        self.webcam_label.pack(pady=20, padx=20)

        # Obszar (Canvas), w którym będzie wyświetlany obraz z kamery.
        self.webcam_canvas = customtkinter.CTkCanvas(self.webcam_frame, width=640, height=480)
        self.webcam_canvas.pack(pady=25, padx=5, anchor="center")

        # Ramka (frame) po prawej stronie, w której umieszczamy wizualizację 3D.
        self.plot_frame = customtkinter.CTkFrame(self.window)
        self.plot_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")

        # Etykieta – nagłówek „Wizualizacja 3D twarzy”.
        self.plot_label = customtkinter.CTkLabel(
            self.plot_frame, text="Wizualizacja 3D twarzy", font=("JetBrains Mono", 40, 'bold')
        )
        self.plot_label.pack(pady=20, padx=20)

        # Tworzenie obiektu Figure z matplotlib i dodawanie do niej osi 3D.
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Tworzenie widgetu FigureCanvasTkAgg – osadzenie matplotlib wewnątrz customtkinter.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Inicjalizacja klasy FacePlotter (odpowiedzialnej za rysowanie twarzy 3D) i uruchomienie animacji.
        self.face_plotter_inst = FacePlotter(
            figure=self.fig,
            axes3d=self.ax,
            canvas=self.canvas,
            root=self.window
        )
        self.face_plotter_inst.start_animation(interval=33)

        # Ramka (frame) do wyświetlania parametrów (po prawej, obok wizualizacji).
        self.params_frame = customtkinter.CTkFrame(self.window)
        self.params_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")

        # Etykieta – nagłówek „Parametry”.
        self.params_title = customtkinter.CTkLabel(
            self.params_frame, text="Parametry:", font=("JetBrains Mono", 40, 'bold')
        )
        self.params_title.pack(pady=20, padx=20)

        # Tekst wyjściowy – inicjalnie przykładowe wartości 0 lub „Brak”.
        params_text = (
            f"MAR:\t\t 0\n"
            f"Ziewanie:\t\t Brak\n"
            f"Licznik ziewnięć:\t 0\n"
            f"Roll:\t\t 0\n"
            f"Pitch:\t\t 0\n"
            f"EAR:\t\t 0\n"
            f"PERCLOS:\t 0%\n"
            f"\n"
            f"FPS:\t\t 0\n"
        )

        # Etykieta wyświetlająca wartości parametrów; używamy monospaced font (JetBrains Mono) i lewego justowania.
        self.params_label = customtkinter.CTkLabel(
            self.params_frame,
            text=params_text,
            font=("JetBrains Mono", 22),
            justify="left",
            width=300
        )
        self.params_label.pack(pady=20, padx=20)

        # Uruchomienie metod, które cyklicznie aktualizują obraz z kamery i parametry w GUI.
        self.update_webcam()
        self.update_labels()


    def show_info_window(self):
        info_window = customtkinter.CTkToplevel(self.window)
        info_window.title("O programie")
        info_window.geometry("500x380")

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

    def queue_image(self, image):
        self.image_queue.put(image)

    def get_face_plotter(self):
        return self.face_plotter_inst

    def set_face_plotter(self, face_plotter_inst):
        self.face_plotter_inst = face_plotter_inst

    def queue_parameters(self, mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps):
        self.data_queue.put((mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps))

    def change_appearance(self, new_appearance_mode):
        mode_dict = {"Jasny": "Light", "Ciemny": "Dark", "Systemowy": "System"}
        customtkinter.set_appearance_mode(mode_dict[new_appearance_mode])

        if mode_dict[new_appearance_mode] == "Light":
            bg_image_path = self.current_path + "/white_bg_img.png"
        elif mode_dict[new_appearance_mode] == "Dark":
            bg_image_path = self.current_path + "/gray_bg_img.jpeg"
        else:
            bg_image_path = self.current_path + "/gray_bg_img.jpeg"

        # Update the background image
        self.bg_image = customtkinter.CTkImage(Image.open(bg_image_path), size=(self.width, self.height))
        self.bg_image_label.configure(image=self.bg_image)

    def update_webcam(self):
        try:
            while not self.image_queue.empty():
                image = self.image_queue.get_nowait()
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.current_image = ImageTk.PhotoImage(image)
                self.webcam_canvas.create_image(0, 0, image=self.current_image, anchor="nw")

        except queue.Empty:
            pass

        if self.running:
            self.window.after(33, self.update_webcam)

    def update_labels(self):
        try:
            while not self.data_queue.empty():
                mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps = self.data_queue.get_nowait()
                self.params_label.configure(
                    text=(
                        f"MAR:\t\t {round(mar, 2)}\n"
                        f"Ziewanie:\t\t {'Obecne' if is_yawning else 'Brak'}\n"
                        f"Licznik ziewnięć:\t {round(yawn_counter, 2)}\n"
                        f"Roll:\t\t {round(roll, 2)}\n"
                        f"Pitch:\t\t {round(pitch, 2)}\n"
                        f"EAR:\t\t {round(ear, 2)}\n"
                        f"PERCLOS:\t {round(perclos * 100, 2)}%\n"
                        f"\n"
                        f"FPS:\t\t {int(fps)}\n"
                    )
                )
        except queue.Empty:
            pass

        if self.running:
            self.window.after(100, self.update_labels)

    def start(self):
        self.window.mainloop()
