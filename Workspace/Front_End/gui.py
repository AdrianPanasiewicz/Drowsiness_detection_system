import customtkinter
import queue

class GUI:
    def __init__(self):
        self.running = True
        self.data_queue = queue.Queue()

        self.initialize_gui()

        self.frame = customtkinter.CTkFrame(self.window)
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.params_label = customtkinter.CTkLabel(self.frame, text="Parametry", font=("Arial", 18))
        self.params_label.pack(pady=20, padx=20)

        # Start periodic updates
        self.update_gui()

    def initialize_gui(self):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        self.window = customtkinter.CTk()
        self.window.geometry("800x600")
        self.window.title("Drowsiness Detection System")
        self.window.resizable(False, False)

    def update_parameters(self, mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps):
        self.data_queue.put((mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps))

    def update_gui(self):
        try:
            while not self.data_queue.empty():
                mar, is_yawning, roll, pitch, ear, perclos, yawn_counter, fps = self.data_queue.get_nowait()
                self.params_label.configure(
                    text=(
                        "Parametry:\n"
                        f"MAR: {round(mar, 2)}\n"
                        f"Ziewanie: {'Obecne' if is_yawning else 'Brak'}\n"
                        f"Roll: {round(roll, 2)}\n"
                        f"Pitch: {round(pitch, 2)}\n"
                        f"EAR: {round(ear, 2)}\n"
                        f"PERCLOS: {round(perclos * 100, 2)}%\n"
                        f"Licznik ziewnięć: {round(yawn_counter, 2)}\n"
                        f"\n"
                        f"FPS: {int(fps)}\n"
                    )
                )
        except queue.Empty:
            pass

        if self.running:
            self.window.after(100, self.update_gui)

    def start(self):
        self.window.mainloop()
