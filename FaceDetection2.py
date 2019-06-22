#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:12:15 2019

@author: jiayungyap
"""
import cv2
import pickle

def nothing(x):
    pass

def drawText(frame, name, x, y):
    cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
def drawRectangle(frame, rect):
    cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

def createTrackbar():
    cv2.namedWindow("Frame")
    cv2.createTrackbar("Scale", "Frame", 0, 5, nothing)
    cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("trainner.yml2")

createTrackbar()

while True:
    ret, frame = cap.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    scale = cv2.getTrackbarPos("Scale", "Frame")
    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")
    
    faces = face_cascade.detectMultiScale(gray, (scale+11)/10, neighbours)
    
    for rect in faces:
        (x, y, w, h) = rect
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(cv2.resize(roi, (300, 300)))
         
        if conf <=30000:
            print(id_)
            print(labels[id_])           
            name = labels[id_]         
        else:
            name = "unknown"
        
        drawText(frame, name, x, y)
        drawRectangle(frame, rect)

    image = cv2.resize(frame,(750, 500))
    cv2.imshow("Frame", image)
    
    key = cv2.waitKey(1)
    
    if key ==27:
        break
    
id_, conf = recognizer.predict(cv2.resize(roi, (300, 300)))
print(conf) 
  
cap.release()
cv2.destroyAllWindows()