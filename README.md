# 🌸 Flower Detection

A deep learning application that detects and classifies flowers using MobileNetV2 and TensorFlow.

## Classes

- 🌼 Daisy (Margaretă)
- 🌻 Dandelion (Păpădie)
- 🌹 Rose (Trandafir)
- 🌻 Sunflower (Floarea soarelui)
- 🌷 Tulip (Lalea)

## Results

- **Train Accuracy:** 99.45%
- **Validation Accuracy:** 88.64%
- **Dataset:** 4317 images, 5 classes

## How it works

1. Dataset loaded from Kaggle (Flowers Recognition)
2. MobileNetV2 pre-trained model (Transfer Learning)
3. Fine-tuned with 10 epochs
4. Tkinter GUI for easy image selection

## Installation

```bash
pip install tensorflow opencv-python numpy matplotlib Pillow
```

## Usage

```bash
# Train the model
python train.py

# Run the GUI app
python app.py
```

## Tech Stack

- Python 3.11
- TensorFlow 2.21
- OpenCV
- Tkinter
