#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:35:56 2019

@author: jiayungyap
"""

import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(
                path)).replace(" ", "-").lower()
            print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            print(label_ids)

            cv2.imshow("Training on image...", cv2.resize(
                cv2.imread(path), (600, 500)))
            cv2.waitKey(100)

            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, dtype="uint8")

            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.4, minNeighbors=3)

            for(x, y, w, h) in faces:
                feature = ()
                roi = image_array[y:y+h, x:x+w]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(roi)
                # res = np.hstack((roi, equalized))
                # cv2.imshow("Training on image...", cv2.resize(
                #     res, (600, 500)))
                # cv2.waitKey(1000)
                x_train.append(cv2.resize(equalized, (300, 300)))
                y_labels.append(id_)

# print(y_labels)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml2")

print("Training Complete")
