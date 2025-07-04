import pathlib


class Config:
    MODES = ["camera", "image", "video"]
    PROCESSING_MODES = ["training", "evaluation",
                        "apply_drowsiness"]

    def __init__(self):
        self.mode = "video"
        self.processing_mode = "apply_drowsiness"

        # Path configuration
        self.base_path = pathlib.Path(
            r"E:\Zycie\Nauka\Studia\Dod\Artykuł naukowy TCNN\NTHUDDD")
        self.training_folder = self.base_path / "Training Dataset"
        self.validation_folder = self.base_path / "Evaluation Dataset"
        self.output_folder = self.base_path / "Processed_dataset"

        # File names
        self.results_name = "results.csv"
        self.training_name = "training_data.csv"
        self.validating_name = "validating_data.csv"
        self.testing_name = "testing_data.csv"

        # Thresholds
        self.perclos_threshold = 0.3
        self.yawn_threshold = 0.5