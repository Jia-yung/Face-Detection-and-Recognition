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

#creates the cascade classification from the haarcascade_frontalface classification file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#creates the facial recogniser using opencv LBPH 
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

#loop through the files to train the recogniser 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        
        #only train with files that ends with png or jpg format
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

            #display the training images
            cv2.imshow("Training on image...", cv2.resize(
                cv2.imread(path), (600, 500)))
            cv2.waitKey(100)

            pil_image = Image.open(path).convert("L")
            #size = (550, 550)
            #final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, dtype="uint8")
            
            #store the detected face in faces array using HaarCascade
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.3, minNeighbors=5)

            #get the x,y position and width, height of the detected face
            for(x, y, w, h) in faces:
                #feature = ()
                roi = image_array[y:y+h, x:x+w]
                #append the faces for training into x_train and their respective lables into y_labels
                x_train.append(roi)
                y_labels.append(id_)

#using pickle to save the labels ID
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

#use the train function to extract features in facial images and converts them to histogram
recognizer.train(x_train, np.array(y_labels))

#save the feature values into a yml file for comparison with input images in recognition process
recognizer.save("trainner.yml2")

print("Training Complete")
