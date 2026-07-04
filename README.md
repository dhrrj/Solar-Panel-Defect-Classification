# Solar Panel Defect Detection

## Description
This project is a Streamlit web application for detecting and classifying defects in solar panel images. It uses a trained deep learning model to predict whether an uploaded panel image belongs to a specific defect category such as Bird Drop, Dusty, Electrical Damage, Physical Damage, Snow Covered, or Clean.

## Installation
1. Clone or download this project folder.
2. Open the project directory in your terminal.
3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the local URL shown in the terminal in your browser.
3. Upload an image of a solar panel.
4. The app will display the uploaded image and show the predicted defect class with confidence scores.

## Contributing
Contributions are welcome. If you would like to improve the project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and test them.
4. Submit a pull request with a clear description of your updates.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.
