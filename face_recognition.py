#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:12:15 2019

@author: jiayungyap
"""
import cv2
import pickle
from preprocessing import TanTriggsPreprocessing
from collections import Counter


def nothing(x):
    pass


def drawText(frame, name, x, y):
    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)


def drawRectangle(frame, rect):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def createTrackbar():
    cv2.namedWindow("Frame")
    cv2.createTrackbar("Scale", "Frame", 0, 5, nothing)
    cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)


class FaceRecognition:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainner.yml2")
        self.labels = {}
        self.read_labels()
        self.count = 0
        self.people = []
        self.temporary = []
        self.toBeRemove = []
        self.Removed = []

    def read_labels(self):
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
            self.labels.update(labels)

    def start_interactive_session(self):
        cap = cv2.VideoCapture(0)
        createTrackbar()
        #preprocessing_algo = TanTriggsPreprocessing()

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scale = cv2.getTrackbarPos("Scale", "Frame")
            neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

            faces = self.face_cascade.detectMultiScale(
                gray, (scale+11)/10, neighbours)

            for rect in faces:
                (x, y, w, h) = rect
                roi = gray[y:y+h, x:x+w]
                # roi = cv2.equalizeHist(roi)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi = clahe.apply(roi)
                # cv2.imshow('show', roi)
                # cv2.waitKey(1000)
                id_, conf = self.recognizer.predict(
                    cv2.resize(roi, (300, 300)))

                if conf <= 55:
                   
                    print(id_)
                    print(self.labels[id_])
                    name = self.labels[id_]
                    print("%s %s" %(name, conf))
                    self.temporary.append(name)
                else:
                    name = "unknown"
                    print("Unknown", conf)
                    
                drawText(frame, name, x, y)
                drawRectangle(frame, rect)
                    
            image = cv2.resize(frame, (750, 500))
            cv2.imshow("Frame", image)

            key = cv2.waitKey(1)

            if key == 27:
                break

        id_, conf = self.recognizer.predict(cv2.resize(roi, (300, 300)))
        print(conf)
        
        
#        for name in self.people:
#            if (self.people.count(name)/len(self.people)) < 0.5:
#                self.toBeRemove.append(name)
#             
#                
#        self.Removed = [x for x in self.people if x not in self.toBeRemove]
        
        for name in self.temporary:
            if name not in self.people:
                self.people.append(name)
        print("People who present", self.people)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition = FaceRecognition()
    recognition.start_interactive_session()
