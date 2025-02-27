# Coconut Detection with YOLOv8

This project uses the YOLOv8 model to detect mature and immature coconuts in real-time using a webcam. The detections are saved as images with timestamps.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/coconut-detection.git
    cd coconut-detection
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 model weights and place them in the project directory:
    ```sh
    # Example command to download the model
    wget https://path/to/your/best1.pt -O best1.pt
    ```

## Usage

1. Run the `coco.py` script to start the coconut detection:
    ```sh
    python coco.py
    ```

2. The script will open a window displaying the webcam feed with detected coconuts highlighted. Detected coconuts will be saved as images with timestamps.

## Configuration

- **Model Path**: Update the path to your YOLOv8 model weights in `coco.py`:
    ```python
    model = YOLO(r'best1.pt')
    ```

- **Custom Class Names**: If you have custom class names, update the `data.yaml` file:
    ```yaml
    names:
      0: Mature Coconut
      1: Immature Coconut
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.