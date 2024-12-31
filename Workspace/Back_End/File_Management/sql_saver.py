import pathlib
import pandas as pd

class SqlSaver():
    default_saving_path = r'Results\results.csv'
    def __init__(self):
        self.working_directory =  pathlib.Path(__file__).parent.parent.parent
        self.saving_path = self.working_directory / self.default_saving_path
        self.index = 0
        self.change_name_if_exists()

    def save_to_csv(self, data: dict) -> None:
        """
        Zapisuje dane do pliku CSV. Jeśli plik istnieje, dane są do niego dopisywane;
        w przeciwnym razie tworzony jest nowy plik, do którego zapisywane są dane.
        Funkcja nie zwraca żadnej wartości i działa wyłącznie jako mechanizm zapisu
        dla dostarczonych danych.

        :param data: Słownik zawierający dane przeznaczone do zapisania w pliku CSV.
        :type data: dict
        :return: None
        :rtype: None
        """
        df = pd.DataFrame(data, index = [self.index])
        if self.saving_path.exists():
            df.to_csv(self.saving_path, mode='a', header=False)
            self.index += 1
        else:
            df.to_csv(self.saving_path)

    def change_name_if_exists(self) -> None:
        """
        Modyfikuje nazwę pliku, aby upewnić się, że nie nadpisze istniejącego pliku. Metoda sprawdza, czy plik już
        istnieje pod podaną ścieżką, a jeśli tak, dodaje unikalny przyrostek w formacie „(i)” do nazwy pliku, gdzie i
        jest rosnącą liczbą całkowitą. Proces ten trwa, aż zostanie znaleziono nazwa bez konfliktów.

        :return: None
        :rtype: None
        """
        i = 1
        original_path = self.saving_path
        while self.saving_path.exists():
            self.saving_path = original_path.with_name(f"{original_path.stem}({i}){original_path.suffix}")
            i += 1


