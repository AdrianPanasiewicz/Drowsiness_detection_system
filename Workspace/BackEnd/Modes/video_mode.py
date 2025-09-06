import re
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Workspace.BackEnd.FileManagement.data_saver import DataSaver


class VideoProcessor:
    def __init__(self, image_processor, perclos_finder,
                 yawn_finder, face_angle_finder,
                 classifier):
        self.image_processor = image_processor
        self.perclos_finder = perclos_finder
        self.yawn_finder = yawn_finder
        self.face_angle_finder = face_angle_finder
        self.classifier = classifier

    def process_video(self, video_path, output_folder,
                      mode, dataset):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(
                f"Error opening video: {video_path}")

        filename = self._generate_filename(video_path, mode, dataset)
        data_saver = DataSaver(filename,
                               save_path=output_folder)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self._reset_finders()

        with tqdm(total=total_frames,
                  desc="Processing frames") as pbar:
            for frame_count in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_frame(frame, frame_count, fps,
                                    data_saver)
                if frame_count % 1000 == 0:
                    pbar.update(1000)

        cap.release()
        data_saver.flush_batch()
        return data_saver.saving_path

    def _process_frame(self, frame, frame_count, fps,
                       data_saver):
        processed_frame, face_mesh_coords = self.image_processor.process_face_image(
            frame)

        if not face_mesh_coords:
            return

        perclos, ear = self.perclos_finder.find_parameter(
            face_mesh_coords)
        is_jawning, yawn_counter, mar = self.yawn_finder.find_parameter(
            face_mesh_coords)
        roll, pitch = self.face_angle_finder.find_parameter(
            face_mesh_coords)

        prediction = self._calculate_prediction(perclos,
                                                mar, ear,
                                                roll, pitch)

        packet = {
            "Frame": frame_count,
            "Timestamp": frame_count / fps,
            "MAR": mar,
            "Roll": roll,
            "Pitch": pitch,
            "EAR": ear,
            "PERCLOS": perclos,
            "Drowsy": prediction
        }
        data_saver.add_to_batch(packet)

    def _calculate_prediction(self, perclos, mar, ear, roll,
                              pitch):
        if perclos >= 0.25:
            return True
        elif 0.125 <= perclos < 0.25:
            data = pd.DataFrame([[mar, ear, roll, pitch]],
                                columns=["MAR", "EAR",
                                         "Roll", "Pitch"])
            return self.classifier.moving_mode_value_prediction(
                data)
        return False

    def _generate_filename(self, video_path, mode, dataset):
        video_path = str(video_path)
        if dataset == "nthuddd":
            if mode == "evaluation":
                return re.search(r'([^\\]+)\.mp4$',
                                 video_path).group(1) + ".csv"
            elif mode == "training":
                match = re.search(
                    r'\\(\d+)\\([^\\]+)\\([^\\]+)\.avi$',
                    video_path)
                if match:
                    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}.csv"
            else:
                raise ValueError("Invalid video mode")

        if dataset == "drozy":
            return re.search(r'([^\\]+)\.mp4$',
                             video_path).group(1) + ".csv"
        else:
            raise ValueError("Invalid dataset")

    def _reset_finders(self):
        self.perclos_finder.reset_memory()
        self.yawn_finder.reset_memory()
        self.face_angle_finder.reset_memory()


class DrowsinessLabelApplier:
    def __init__(self, label_source, output_folder):
        self.label_source = Path(label_source)
        self.output_folder = Path(output_folder)

    def apply_labels_from_folders(self):
        csv_files = list(self.output_folder.glob('*.csv'))

        for i, csv_file in enumerate(csv_files, 1):
            match = re.search(
                r"(\d+)_([^_]+(?:_[^_]+)*)_([^_]+)\.csv$",
                csv_file.name)
            if not match:
                continue

            txt_path = self.label_source / match.group(
                1) / match.group(
                2) / f"{match.group(1)}_{match.group(3)}_drowsiness.txt"
            self._apply_label_from_txt(csv_file, txt_path)
            print(f"Processed {i}/{len(csv_files)}: {csv_file.name}")

    def apply_labels_from_txt_matrix(self):
        kss_file = open(self.label_source)
        labels = kss_file.read().split()
        # Access it by (x-1)*3 + (y-1)
        csv_files = list(self.output_folder.glob('*.csv'))

        for i,file in enumerate(csv_files,1):
            numbers = list(map(int, re.findall(r'\d+', file.name)))
            index = (numbers[0]-1)*3 + numbers[1] - 1
            kss_scale = int(labels[index])
            if kss_scale > 5:
                drowsiness_label = 1
            else:
                drowsiness_label = 0

            df = pd.read_csv(file)
            df['Drowsy'] = drowsiness_label
            # df.drop('PERCLOS', axis=1, inplace=True)
            df.to_csv(file, index=False)
            print(f"Processed {i}/{len(csv_files)}: {file.name}")

    def _apply_label_from_txt(self, csv_file, txt_path):
        try:
            with open(txt_path, 'r') as f:
                drowsiness_data = f.read().strip()
        except FileNotFoundError:
            print(f"Label file not found: {txt_path}")
            return

        df = pd.read_csv(csv_file)
        if len(drowsiness_data) != len(df):
            print(f"Label length mismatch in {txt_path}")

        df['Drowsy'] = list(map(int, drowsiness_data.ljust(
            len(df), '0')[:len(df)]))
        df.to_csv(csv_file, index=False)