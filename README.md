# Solar Panel Defect Detection

A Streamlit web application for detecting and classifying defects in solar panel images using a trained deep learning model.

## Overview
This project helps identify whether a solar panel image belongs to one of these categories:
- Bird-drop
- Clean
- Dusty
- Electrical-damage
- Physical-Damage
- Snow-Covered

The app accepts an uploaded image and returns the predicted class along with confidence scores.

## Features
- Upload an image through a web interface
- Preprocess the image automatically
- Run inference using a TensorFlow/Keras model
- Display predicted class and confidence percentages

## Tech Stack
- Python
- Streamlit
- TensorFlow/Keras
- NumPy
- Pillow

## Project Structure
```text
Solar Panel Defect/
├── app.py
├── requirements.txt
├── mobilenetv5.keras
├── efficient_model.keras
├── Data/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
└── solar panel defect.ipynb
```

## Installation
1. Clone or download this repository.
2. Open the project folder in your terminal.
3. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Start the app:

```bash
streamlit run app.py
```

2. Open the local URL shown in the terminal in your browser.
3. Upload an image of a solar panel.
4. Click Predict to view the predicted defect class and confidence scores.

## Model Files
The application looks for trained model files in the project root:
- mobilenetv5.keras
- efficient_model.keras

If neither file is present, the app will show an error message.

## Dataset
The dataset used for training and evaluation is stored in the Data folder. It includes multiple defect categories and a clean class.

## Contributing
Contributions are welcome. If you would like to improve this project:
1. Fork the repository
2. Create a new branch
3. Make your changes and test them
4. Submit a pull request

## License
This project is licensed under the MIT License.
