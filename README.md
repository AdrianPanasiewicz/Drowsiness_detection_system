# Real-Time Drowsiness and Emotion Detection

This project aims to develop a real-time system that detects drowsiness and emotions based on facial data. Designed for applications in driver safety, workplace monitoring, and human-computer interaction, the system uses advanced computer vision techniques and machine learning algorithms to process video feed and analyze facial expressions for signs of drowsiness and specific emotions.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Drowsiness Detection**: Detects signs of fatigue or drowsiness in real-time using facial landmarks and eye-blink rates.
- **Emotion Recognition**: Identifies emotions such as happiness, sadness, anger, surprise, etc.
- **Live Video Feed Processing**: Processes live video feeds from webcams or IP cameras.
- **Alerts**: Configurable alerts for specific thresholds (e.g., when drowsiness is detected).
- **Data Logging**: Records detection events for further analysis and review.

## Technologies Used
- **Python**
- **OpenCV**: For real-time image processing and face detection.
- **Mediapipe**: For facial landmark detection.
- **Fastai**: Used for emotion detection with a CNN model.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/AdrianPanasiewicz/Real-time-drowsiness-and-emotion-detection.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Connect your IP camera or configure your webcam for live feed.

## Usage
1. Run the main script to start the detection system:
    ```bash
    python main.py
    ```

## Project Structure
- **main.py**: Main script to initialize and run the detection system.
- **config.yaml**: Configuration file for camera settings, alert thresholds, and system options.
- **/Back_End**: Classes for managing the flow and processing of information.
- **/Front_End**: GUI responsible for showcasing the results
- **/Resources**: Contains pre-trained models for emotion detection.
- **/Utilities**: Helper functions for data processing and logging.


## Examples
![Drowsiness Detection Example](examples/drowsiness_example.jpg)

![Emotion Detection Example](examples/emotion_example.jpg)

## Contributing
Feel free to open issues or pull requests. Contributions are welcome to improve detection accuracy, add new features, and optimize performance.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
