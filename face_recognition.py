#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:12:15 2019

@author: jiayungyap
"""

import cv2
import pickle
from math import sin, cos, radians

def nothing(x):
    pass

#function to write the name of the detected faces.
def drawText(frame, name, x, y):
    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)

#function to draw a green bounding rectangle of the detected faces.
def drawRectangle(frame, rect):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#trackbar used for adjusting the scale factor and min neighors for face detection using Haar Cascade
def createTrackbar():
    cv2.namedWindow("Frame")
    cv2.createTrackbar("Scale", "Frame", 5, 20, nothing)
    cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

#rotates the image array by  angle degree
def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]

class FaceRecognition:
     
    def __init__(self):
        
        #creates the cascade classification from the haarcascade_frontalface classification file
        self.face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        #read the file containing all the trained datasets of individuals
        self.recognizer.read("trainner.yml2")
        
        #initialise the variables
        self.labels = {}
        self.read_labels()
        self.count = 0
        self.people = []
        self.temporary = []
    
    def read_labels(self):
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
            self.labels.update(labels)

    def start_interactive_session_images(self):        
        
        #read the input image use for face detection and recognition
        img = cv2.imread("brightClass.jpg")
        
        #apply gaussian blur to remove noises
        blurredImg = cv2.GaussianBlur(img, (3, 3), 0)
        
        #convert image from rgb to gray scale
        gray = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2GRAY)
        
        #to detect for faces in certain angle
        for angle in [0, -25, 25]:
            
            #perform image transformation of rotated face                                                
            rimg = rotate_image(gray, angle)
            
            #using haarcascade to detect for the presence of face
            #store the detected face in faces array
            faces = self.face_cascade.detectMultiScale(rimg, 1.5, 5)
            
            #loop though all the face in faces array
            for face in faces:
                rotatedFace = [rotate_point(face, gray, -angle)]
                       
                for index, rect in enumerate(rotatedFace):
                    #get the x,y position and width, height of the detected face
                    (x, y, w, h) = rect
                    
                    #extract the region of interest where the face is detected using x, y, w, h values
                    roi = gray[y:y+h, x:x+w]
                  
                    #use LBPH to predict face within the region of interest
                    #returns a confidence level of how similar the input face to the faces in the trained dataset
                    #returns an ID of the trained individual
                    id_, conf = self.recognizer.predict(roi)
                    
                    #set a threshold for the distance between the input images and trained images in database
                    #if the difference in terms of distance is less than the threshold
                    #assign the name of the person to the one that have the closest match in the trained datatset
                    #0 indicates a perfect match
                    if conf <= 50:
                        
                        print(id_)
                        print(self.labels[id_])
                        name = self.labels[id_]
                        print("%s %s" %(name, conf))
                        
                        #assign the name of the recognised individual to an array to keep track of the people present in class
                        self.temporary.append(name)
                    else:
                        #assign unknown to name if the difference between the compared images are above threhold level
                        name = "unknown"
                        print("Unknown", conf)
                     
                    #write the name of the individual at the coordinate where the face is detected
                    drawText(img, name, x, y)
                    
                    #draw a bounding rectangle around the detected face
                    drawRectangle(img, rect)
                
            #display the output image            
            image = cv2.resize(img, (1024, 640))
            cv2.imshow("Frame", image)
            
            print("People who present", self.temporary)
            key = cv2.waitKey(1)            
            if key == 27:
                break            
    
    #function for video capture
    #face detection and recognition procedure is same as start_interactive_session_images
    def start_interactive_session(self):
        cap = cv2.VideoCapture(0)
        createTrackbar()        
        
        while True:
            ret, frame = cap.read()
            
            blurredImg = cv2.GaussianBlur(frame, (3, 3), 0)
            
            gray = cv2.cvtColor(blurredImg, cv2.COLOR_BGR2GRAY)
            
            scale = cv2.getTrackbarPos("Scale", "Frame")
            neighbours = cv2.getTrackbarPos("Neighbours", "Frame")
            
            for angle in [0, -25, 25]:
                rimg = rotate_image(gray, angle)
                faces = self.face_cascade.detectMultiScale(rimg, (scale+11)/10, neighbours)
                
            
                for face in faces:
                    rotatedFace = [rotate_point(face, gray, -angle)]
                           
                    for index, rect in enumerate(rotatedFace):
                        (x, y, w, h) = rect
                        roi = gray[y:y+h, x:x+w]
                      
                        id_, conf = self.recognizer.predict(roi)
                        
                        if conf <= 30:
                           
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
                        
        for name in self.temporary:
            if name not in self.people:
                self.people.append(name)
        print("People who present", self.people)

        cap.release()
        cv2.destroyAllWindows()

#main program to start the execution of code
if __name__ == '__main__':
    recognition = FaceRecognition()
    #recognition.start_interactive_session()
    recognition.start_interactive_session_images()
