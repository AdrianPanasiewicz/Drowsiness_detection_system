import pandas as pd
import math


class DatasetCreator:
    default_name = "results.csv"

    def __init__(self, load_folder, save_path):
        self.raw_data = pd.DataFrame()
        self.load_folder = load_folder
        self.save_path = save_path
        self.global_sequence_counter = 0

    def process_data(self, sequence_length):
        all_csv_files = [f for f in self.load_folder.rglob('*') if
                  f.is_file() and f.suffix in '.csv']

        processed_data = pd.DataFrame()

        for csv in all_csv_files:
            loaded_data = pd.read_csv(csv)
            sequences = self.create_sequences_from_data(sequence_length, loaded_data)
            processed_data = pd.concat([processed_data,sequences], ignore_index=True)

        if self.save_path.exists():
            processed_data.to_csv(self.save_path, mode='a',
                      header=False, index=False)
        else:
            processed_data.to_csv(self.save_path, index=False)

    def create_sequences_from_data(self, length, df):
        output_data = pd.DataFrame()
        total_length = df.shape[0]
        for i in range(0,math.floor(total_length/length)):
            sequence = df.loc[i*length:(i+1)*length - 1, ["Frame", "MAR", "Roll", "Pitch", "EAR", "Drowsy"]]
            sequence_with_id = sequence.copy()
            sequence_with_id['Sequence_id'] = self.global_sequence_counter
            output_data = pd.concat([output_data,sequence_with_id],ignore_index=True)
            self.global_sequence_counter += 1

        return output_data
