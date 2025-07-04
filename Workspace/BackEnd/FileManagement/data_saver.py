import pathlib
import pandas as pd
import openpyxl
from typing import Dict, Any, List
import threading
import time

class DataSaver:
    """
    Klasa DataSaver zapewnia funkcje umożliwiające zapisywanie danych w formacie CSV,
    unikając przy tym nadpisania istniejących plików. Główne metody to:
    - save_to_csv(): dopisywanie danych do istniejącego pliku bądź tworzenie nowego.
    - change_name_if_exists(): automatyczne zmienianie nazwy pliku, jeśli plik o danej nazwie istnieje.
    """

    default_filename = "results.csv"

    def __init__(self, filename: str = default_filename, save_path: pathlib.Path = None,
                 batch_size: int = 100, auto_flush_interval: float = 30.0,
                 auto_flush_enabled: bool = False, flush_thread = None) -> None:
        """
        Inicjalizuje obiekt DataSaver, ustalając ścieżkę docelową pliku CSV i
        zapewniając unikalną nazwę pliku (jeżeli istnieje konflikt nazw).
        """
        # Ustalanie ścieżki domyślnej, wychodząc z lokalizacji bieżącego pliku.
        if save_path is None:
            self.working_directory = pathlib.Path(__file__).parent.parent.parent
            relative_saving_path = fr'Results\{filename}'
            self.saving_path= self.working_directory / relative_saving_path
            self.saving_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.saving_path = pathlib.Path(save_path) / filename
        self.index = 0
        self.change_name_if_exists()

        self.batch_size = batch_size
        self.auto_flush_interval = auto_flush_interval
        self.batch_buffer = []
        self.last_flush_time = time.time()
        self.lock = threading.Lock()
        self.auto_flush_enabled = auto_flush_enabled
        self.flush_thread = flush_thread

    def save_to_csv(self, data: Dict[str, Any]) -> None:
        """
        Zapisuje słownik 'data' do pliku CSV. Jeśli plik już istnieje, dane są dopisywane;
        w przeciwnym razie tworzony jest nowy plik.

        :param data: Słownik z danymi do zapisania w pliku CSV.
        :type data: dict
        :rtype: None
        """
        df = pd.DataFrame(data, index=[self.index])

        if self.saving_path.exists():
            df.to_csv(self.saving_path, mode='a', header=False)
            self.index += 1
        else:
            df.to_csv(self.saving_path)

    def add_to_batch(self, data: Dict[str, Any]) -> None:
        """
        Dodaje dane do batcha. Automatycznie zapisuje batch gdy osiągnie zadany rozmiar.

        :param data: Słownik z danymi do dodania do batcha
        """
        with self.lock:
            self.batch_buffer.append(data.copy())


            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

    def batch_save_to_csv(self, data_list: List[
        Dict[str, Any]]) -> None:
        """
        Zapisuje listę słowników do pliku CSV w jednej operacji.

        :param data_list: Lista słowników z danymi do zapisania
        """
        if not data_list:
            return

        df = pd.DataFrame(data_list)

        # Jeśli plik istnieje, dopisz dane. W przeciwnym razie utwórz nowy plik.
        if self.saving_path.exists():
            df.to_csv(self.saving_path, mode='a',
                      header=False, index=False)
        else:
            df.to_csv(self.saving_path, index=False)

        self.index += len(data_list)

    def _flush_batch(self) -> None:
        """
        Wewnętrzna metoda do zapisywania batcha. Nie używać bezpośrednio.
        """
        if self.batch_buffer:
            self.batch_save_to_csv(self.batch_buffer)
            self.batch_buffer.clear()
            self.last_flush_time = time.time()

    def flush_batch(self) -> None:
        """
        Wymusza natychmiastowe zapisanie wszystkich danych z batcha.
        """
        with self.lock:
            self._flush_batch()

    def start_auto_flush(self) -> None:
        """
        Rozpoczyna automatyczne zapisywanie batcha w określonych interwałach.
        """
        if not self.auto_flush_enabled:
            self.auto_flush_enabled = True
            self.flush_thread = threading.Thread(
                target=self._auto_flush_worker, daemon=True)
            self.flush_thread.start()

    def stop_auto_flush(self) -> None:
        """
        Zatrzymuje automatyczne zapisywanie batcha.
        """
        self.auto_flush_enabled = False
        if self.flush_thread:
            self.flush_thread.join(timeout=1.0)
        self.flush_batch()  # Final flush

    def _auto_flush_worker(self) -> None:
        """
        Wątek roboczy dla automatycznego zapisywania batcha.
        """
        while self.auto_flush_enabled:
            time.sleep(1.0)

            with self.lock:
                if (
                        time.time() - self.last_flush_time >= self.auto_flush_interval and
                        self.batch_buffer):
                    self._flush_batch()

    def get_batch_status(self) -> Dict[str, Any]:
        """
        Zwraca informacje o stanie batcha.

        :return: Słownik z informacjami o stanie batcha
        """
        with self.lock:
            return {
                "batch_size": len(self.batch_buffer),
                "max_batch_size": self.batch_size,
                "auto_flush_enabled": self.auto_flush_enabled,
                "time_since_last_flush": time.time() - self.last_flush_time,
                "auto_flush_interval": self.auto_flush_interval
            }

    def __enter__(self):
        self.start_auto_flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_auto_flush()

    def save_to_excel(self, data: Dict[str, Any]) -> None:
        """
        Zapisuje słownik 'data' do pliku CSV. Jeśli plik już istnieje, dane są dopisywane;
        w przeciwnym razie tworzony jest nowy plik.

        :param data: Słownik z danymi do zapisania w pliku CSV.
        :type data: dict
        :rtype: None
        """

        df = pd.DataFrame(data, index=[self.index])
        excel_path = self.saving_path.with_suffix(".xlsx")

        if excel_path.exists():
            with pd.ExcelWriter(excel_path,
                                engine="openpyxl", mode="a",
                                if_sheet_exists="overlay") as writer:
                # Get current number of rows to place new data correctly
                workbook = openpyxl.load_workbook(
                    excel_path)
                sheet = workbook.active
                start_row = sheet.max_row + 1
                df.to_excel(writer, index=False,
                            header=False,
                            startrow=start_row - 1)
        else:
            df.to_excel(excel_path, index=False)

    def change_name_if_exists(self) -> None:
        """
        Zapobiega nadpisaniu istniejących plików, dodając przyrostek (i) do nazwy pliku
        w przypadku wykrycia kolizji nazw. Wartość 'i' jest inkrementowana do momentu
        znalezienia wolnej nazwy pliku.

        :rtype: None
        """
        i = 1
        original_path = self.saving_path
        while self.saving_path.exists():
            # Dodanie przyrostka (i) do nazwy istniejącego pliku i sprawdzenie,
            # czy nowa nazwa jest dostępna.
            self.saving_path = original_path.with_name(
                f"{original_path.stem}({i}){original_path.suffix}"
            )
            i += 1
