# Real-Time drowsiness detection system

This project is an implementation of a real-time system that detects drowsiness based on facial data. Designed for applications in driver safety, the system uses computer vision techniques and machine learning algorithms to process video feed and analyze facial expressions for signs of drowsiness.

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
- **Live Video Feed Processing**: Processes live video feeds from webcams or IP cameras.
- **Alerts**: Configurable alerts for specific thresholds (e.g., when drowsiness is detected).
- **Data Logging**: Records detection events for further analysis and review.

## Technologies Used
- **Python**
- **OpenCV**: For real-time image processing and face detection.
- **Mediapipe**: For facial landmark detection.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/AdrianPanasiewicz/Drowsiness_detection_system.git
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
- **/Results**: Results of the program saved in csv files.
- **/Utilities**: Helper functions for data processing and logging.


## Examples
![Drowsiness Detection Example](examples/drowsiness_example.jpg)

![Emotion Detection Example](examples/emotion_example.jpg)

## Contributing
Feel free to open issues or pull requests. Contributions are welcome to improve detection accuracy, add new features, and optimize performance.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
