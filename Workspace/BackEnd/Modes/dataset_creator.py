import pandas
import pandas as pd


class DatasetCreator:
    default_name = "results.csv"

    def __init__(self, data_saver, load_folder, save_folder, output_name = default_name):
        self.data_saver = data_saver
        self.raw_data = pd.DataFrame()
        self.load_folder = load_folder
        self.save_folder = save_folder
        self.output_name = output_name

    def load_data(self):
        all_csv_files = [f for f in self.load_folder.rglob('*') if
                  f.is_file() and f.suffix in '.csv']

        loaded_data = pd.DataFrame()
        processed_data = pd.DataFrame()

        for csv in all_csv_files:
            loaded_data = pd.read_csv(csv)



    def create_sequence_sample(self,length,dataframe):
        pass