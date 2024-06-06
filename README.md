# AI_Saksham_Project
# Driver Behavior Monitoring System

This project aims to enhance road safety by monitoring and analyzing driver behavior in real-time using advanced computer vision and machine learning techniques. The system can detect drowsiness, distraction, and emotional states such as happiness, neutrality, and anger. It provides real-time alerts to the driver to prevent accidents.

## Import Packages

The following packages are used in this project. Ensure you have them installed before running the system.

```python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from ultralytics import YOLO
import math
from keras.models import model_from_json
```


You can install the necessary packages using pip:

```bash
pip install scipy imutils numpy argparse dlib opencv-python ultralytics keras
```

# Main File

The main file for executing the project is tm2.py. This script integrates all the components of the Driver Behavior Monitoring System, including face detection, emotion recognition, drowsiness detection, and mobile phone usage detection.

# How to Run the Project

Follow these steps to set up and run the Driver Behavior Monitoring System:

# Clone the Repository:

```bash
git clone https://github.com/your-repository/driver-behavior-monitoring.git
cd driver-behavior-monitoring
```

# Install Dependencies:

Ensure you have all the required packages installed. You can use the provided requirements.txt file if available:

```bash
pip install -r requirements.txt
```

# Prepare Models:

Download or prepare the necessary machine learning models for emotion recognition and drowsiness detection. Place the model files in the appropriate directories as expected by the tm2.py script.

# Run the Main Script:

Execute the main script tm2.py to start the Driver Behavior Monitoring System.

```bash
python tm2.py
```

# Camera Setup:

Ensure that your webcam or external camera is properly connected and recognized by the system. The script will use the camera to capture real-time video footage of the driver.

# Project Structure

Here's a brief overview of the project structure:

```bash
driver-behavior-monitoring/
│
├── tm2.py                     # Main script for executing the project
├── models/                    # Directory for storing machine learning models
│   ├── emotion_model.json     # Model architecture for emotion recognition
│   ├── emotion_model.h5       # Model weights for emotion recognition
│   └── ...                    # Additional model files
│
├── data/                      # Directory for storing data samples
│   ├── sample_images/         # Example images used for testing
│   └── ...                    # Additional data files
│
├── requirements.txt           # List of required Python packages
├── README.md                  # Project documentation
└── ...                        # Other scripts and files
```

# Notes
Ensure your environment meets all the hardware and software requirements specified.
For best performance, use a high-resolution camera and ensure proper lighting conditions.
The system should be tested in various scenarios to validate its accuracy and reliability.
By following these instructions, you should be able to set up and run the Driver Behavior Monitoring System effectively. If you encounter any issues, please refer to the documentation or seek assistance from the project maintainers.
