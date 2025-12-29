# Gender & Age Detection using OpenCV

## Overview
This project implements a Gender and Age Prediction system using OpenCV and
pretrained deep learning models. The system detects faces from images or webcam
input and predicts the gender and age range of the detected person.

The project is designed for beginners and focuses on inference using pretrained
models rather than training from scratch.

---

## Technologies Used
- Python
- OpenCV
- Deep Learning (Pretrained CNN models)

---

## Features
- Face detection using OpenCV DNN module
- Gender classification (Male / Female)
- Age range estimation
- Static image processing (single output)
- Webcam-based detection (single frame capture)
- Visualized results on image

---

## Project Structure
```

Gender-and-Age-Detection/
│
├── detect.py                    # Image-based detection
├── detect_webcam_single.py      # Webcam single-frame detection
│
├── age_deploy.prototxt
├── age_net.caffemodel
├── gender_deploy.prototxt
├── gender_net.caffemodel
├── opencv_face_detector.pb
├── opencv_face_detector.pbtxt
│
├── outputs
│   ├── output.png
└── README.md

````

---

## Installation
Install required Python libraries using:

```bash
pip install -r requirements.txt
````

---

## How to Run (Image Input)

Place the image inside the project folder or `sample_images` directory and run:

```bash
python detect.py --image sample_images/woman1.jpg
```

---

## How to Run (Webcam Input)

This captures a single frame from the webcam and predicts gender and age once:

```bash
python detect_webcam_single.py
```

---

## Output

* Detected face is highlighted with a bounding box
* Predicted gender and age range are displayed on the image
* Output is shown once (no continuous looping)

---

## Notes

* This project uses pretrained models for prediction
* Accuracy depends on image quality and lighting conditions
* No model training is performed

---

## Author

Supriya Kulkarni
