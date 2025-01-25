import pathlib
import pandas as pd
from typing import Dict, Any

class SqlSaver:
    """
    Klasa SqlSaver zapewnia funkcje umożliwiające zapisywanie danych w formacie CSV,
    unikając przy tym nadpisania istniejących plików. Główne metody to:
    - save_to_csv(): dopisywanie danych do istniejącego pliku bądź tworzenie nowego.
    - change_name_if_exists(): automatyczne zmienianie nazwy pliku, jeśli plik o danej nazwie istnieje.
    """

    default_filename = "results.csv"

    def __init__(self, filename: str = default_filename) -> None:
        """
        Inicjalizuje obiekt SqlSaver, ustalając ścieżkę docelową pliku CSV i
        zapewniając unikalną nazwę pliku (jeżeli istnieje konflikt nazw).
        """
        # Ustalanie ścieżki domyślnej, wychodząc z lokalizacji bieżącego pliku.
        self.working_directory = pathlib.Path(__file__).parent.parent.parent
        relative_saving_path = fr'Results\{filename}'
        self.saving_path= self.working_directory / relative_saving_path
        self.saving_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.change_name_if_exists()

    def save_to_csv(self, data: Dict[str, Any]) -> None:
        """
        Zapisuje słownik 'data' do pliku CSV. Jeśli plik już istnieje, dane są dopisywane;
        w przeciwnym razie tworzony jest nowy plik.

        :param data: Słownik z danymi do zapisania w pliku CSV.
        :type data: dict
        :rtype: None
        """
        df = pd.DataFrame(data, index=[self.index])
        # Jeśli plik istnieje, dopisz dane. W przeciwnym razie utwórz nowy plik.
        if self.saving_path.exists():
            df.to_csv(self.saving_path, mode='a', header=False)
            self.index += 1
        else:
            df.to_csv(self.saving_path)

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
