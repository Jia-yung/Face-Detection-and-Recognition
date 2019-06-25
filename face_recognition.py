#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:12:15 2019

@author: jiayungyap
"""
import cv2
import pickle
from preprocessing import TanTriggsPreprocessing


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
        self.recognizer = cv2.face.EigenFaceRecognizer_create()
        self.recognizer.read("trainner.yml2")
        self.labels = {}
        self.read_labels()

    def read_labels(self):
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
            self.labels.update(labels)

    def start_interactive_session(self):
        cap = cv2.VideoCapture(0)
        createTrackbar()
        preprocessing_algo = TanTriggsPreprocessing()

        while True:
            ret, frame = cap.read()
            gray = frame
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scale = cv2.getTrackbarPos("Scale", "Frame")
            neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

            faces = self.face_cascade.detectMultiScale(
                gray, (scale+11)/10, neighbours)

            for rect in faces:
                (x, y, w, h) = rect
                roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                roi = cv2.equalizeHist(roi)
                # print(roi.shape)
                cv2.imshow('show', roi)
                id_, conf = self.recognizer.predict(
                    cv2.resize(roi, (300, 300)))

                if conf <= 30000:
                    print(id_)
                    print(self.labels[id_])
                    name = self.labels[id_]
                else:
                    name = "unknown"

                drawText(frame, name, x, y)
                drawRectangle(frame, rect)

            image = cv2.resize(frame, (750, 500))
            cv2.imshow("Frame", image)

            key = cv2.waitKey(1)

            if key == 27:
                break

        id_, conf = self.recognizer.predict(cv2.resize(roi, (300, 300)))
        print(conf)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    recognition = FaceRecognition()
    recognition.start_interactive_session()
