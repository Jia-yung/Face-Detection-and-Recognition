# Face-Detection-and-Recognition
A face detection and face recognition system using Haar Cascade and Local Binary Pattern Histogram(LBPH) from OpenCV.

![face recognition output](https://github.com/Jia-yung/Face-Detection-and-Recognition/blob/master/output%20images/result.jpg)

## Setup and Installation
The face detection recognition requires OpenCV and OpenCV contrib python. These can be installed using package manager [pip](https://pip.pypa.io/en/stable/) to install
```bash
pip install opencv-python
```
```bash
pip install opencv_contrib_python
```
## Running the face_train
Before we can actually recognise faces, we need to first run the face_train.py file to train the recogniser on the list of faces located in the same directory.

The faceTrain will access the folder with a list of individual faces in their respective folder and recognise each facial feature. 
```bash
python face_train.py
```

## Running the face_recognition
Run the face_recognition.py file to start recognising faces. Recgonised faces will have their name shown above a green bounding box around detected faces and unknown for failed recognition.
```bash
python face_recognition.py
```
