# Biomed Project

## Overview
This directory contains resources and scripts related to the Biomed project. The project focuses on predicting hydration levels using machine learning models and includes data processing, training, and analysis components.

## Directory Structure

```
Biomed/
├── predict_hydration.py       # Script for predicting hydration levels
├── train_model.py             # Script for training the machine learning model
├── requirements.txt           # Dependencies required for the project
├── data/                      # Directory containing data-related files
│   ├── data_logger.py         # Script for logging data
│   ├── gsr.ino                # Arduino script for GSR sensor
```

## Getting Started

### Prerequisites
- Python 3.x
- Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Project
1. Train the model:
   ```bash
   python train_model.py
   ```
2. Predict hydration levels:
   ```bash
   python predict_hydration.py
   ```

## Data
The `data/` directory contains scripts and resources for data collection and preprocessing. The `data_logger.py` script is used for logging data, and `gsr.ino` is an Arduino script for the GSR sensor.

## Collecting GSR Data

To collect Galvanic Skin Response (GSR) data, follow these steps:

1. **Setup the Hardware**:
   - Use the `gsr.ino` Arduino script located in the `data/` directory to program your Arduino board.
   - Connect the GSR sensor to the Arduino board as per the sensor's documentation.

2. **Upload the Script**:
   - Open the `gsr.ino` file in the Arduino IDE.
   - Connect your Arduino board to your computer.
   - Upload the script to the Arduino board.

3. **Start Data Logging**:
   - Run the `data_logger.py` script to log the GSR data:
     ```bash
     python data/data_logger.py
     ```
   - Ensure the Arduino board is connected to the computer and the correct serial port is specified in the script.

The logged data will be saved for further processing and analysis.

## Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## License
This project is licensed under the MIT License.