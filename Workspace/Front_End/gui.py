import customtkinter
import queue
from PIL import Image, ImageTk
import os
import cv2

class GUI:
    def __init__(self):
        self.running = True
        self.data_queue = queue.Queue()
        self.image_queue = queue.Queue()
        self.current_image = None
        self.width = 1400
        self.height = 650

        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.window = customtkinter.CTk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("System wykrywania senności")
        self.window.resizable(False, False)

        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.bg_image = customtkinter.CTkImage(Image.open(self.current_path + "/gray_bg_img.jpeg"),
                                               size=(self.width, self.height))
        self.bg_image_label = customtkinter.CTkLabel(self.window,text = "", image=self.bg_image)
        self.bg_image_label.place(x=0, y=0, relwidth=1, relheight=1)

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

        self.webcam_frame = customtkinter.CTkFrame(self.window)
        self.webcam_frame.pack(pady=20, padx=20, fill="both", expand=True, side="left")
        self.webcam_label = customtkinter.CTkLabel(self.webcam_frame, text="Webcam", font=("Roboto", 40))
        self.webcam_label.pack(pady=20, padx=20)
        self.webcam_canvas = customtkinter.CTkCanvas(self.webcam_frame, width=640, height=480)
        self.webcam_canvas.pack(pady=25, padx=5, anchor="center")

        self.plot_frame = customtkinter.CTkFrame(self.window)
        self.plot_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")
        self.plot_label = customtkinter.CTkLabel(self.plot_frame, text="Plot twarzy w 3d", font=("Roboto", 40))
        self.plot_label.pack(pady=20, padx=20)

        self.params_frame = customtkinter.CTkFrame(self.window)
        self.params_frame.pack(pady=20, padx=20, fill="both", expand=True, side="right")
        self.params_title = customtkinter.CTkLabel(self.params_frame, text="Parametry:", font=("Roboto", 40))
        self.params_title.pack(pady=20, padx=20)
        params_text = (
            "Parametry:\n"
            f"MAR: {0}\n"
            f"Ziewanie: {'Brak'}\n"
            f"Licznik ziewnięć: {0}\n"
            f"Roll: {0}\n"
            f"Pitch: {0}\n"
            f"EAR: {0}\n"
            f"PERCLOS: {0}%\n"
            f"\n"
            f"FPS: {0}\n"
        )
        self.params_label = customtkinter.CTkLabel(self.params_frame, text=params_text, font=("Roboto", 24))
        self.params_label.pack(pady=20, padx=20)

        self.update_webcam()
        self.update_labels()

    def show_info_window(self):
        info_window = customtkinter.CTkToplevel(self.window)
        info_window.title("O programie")
        info_window.geometry("400x300")

        info_window.attributes("-topmost", 1)

        info_label = customtkinter.CTkLabel(
            info_window,
            text="Tutaj możesz napisać, co robi Twój program, jak działa,\n"
                 "i co należy zrobić, by go używać. Dodaj dowolne informacje."
        )
        info_label.pack(pady=20, padx=20)

    def queue_image(self, image):
        self.image_queue.put(image)

    def queue_parameters(self, mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps):
        self.data_queue.put((mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps))

    def change_appearance(self, new_appearance_mode):
        mode_dict = {"Jasny": "Light", "Ciemny": "Dark", " Systemowy": "System"}
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
                        "Parametry:\n"
                        f"MAR: {round(mar, 2)}\n"
                        f"Ziewanie: {'Obecne' if is_yawning else 'Brak'}\n"
                        f"Licznik ziewnięć: {round(yawn_counter, 2)}\n"
                        f"Roll: {round(roll, 2)}\n"
                        f"Pitch: {round(pitch, 2)}\n"
                        f"EAR: {round(ear, 2)}\n"
                        f"PERCLOS: {round(perclos * 100, 2)}%\n"
                        f"\n"
                        f"FPS: {int(fps)}\n"
                    )
                )
        except queue.Empty:
            pass

        if self.running:
            self.window.after(100, self.update_labels)

    def start(self):
        self.window.mainloop()
