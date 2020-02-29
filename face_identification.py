# facial recognition and identification, final program, using the .yml data of the face_machine_learning.py 
# Editor: Philip Schmidt
# Date: 24.01.2020
# Detailed script documentation in form of comments right next to the code

# Enjoy using the program!

# Installing and upgrading all necessary packages
import os, sys
import numpy as np
from subprocess import call

my_packages = ['opencv-contrib-python']
def upgrade(package_list):
    call(['pip', 'install', '--upgrade', '--user'] + package_list)
upgrade(my_packages)

import cv2
import pickle


face_recognition = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')   # .xml data for face_recognition
eye_recognition = cv2.CascadeClassifier('haarcascades\haarcascade_eye.xml')                 # .xml data for eye_recognition

face_identification = cv2.face.LBPHFaceRecognizer_create()                              # local binary patterns histograms
face_identification.read("facetrainer.yml")                                             # machine learning data set

# Access to pickle_folder of machinelearning dataset, to know the name of the person in the dataset
folder_name = {"name_person": 1}
with open("folder_name.pickle", 'rb') as f:  
    original_folder_name = pickle.load(f)
    folder_name = {v:k for k,v in original_folder_name.items()} 

# Decleration of usb-camera  
camera = cv2.VideoCapture(0)

while(True):
    # Camerarecording frame by frame
    ret, record = camera.read()

    # Converting to gray values
    grey = cv2.cvtColor(record, cv2.COLOR_BGR2GRAY)

    # Defining of coordinates from recognized faces
    faces = face_recognition.detectMultiScale( grey, scaleFactor = 2, minNeighbors = 4)     #using of xml_data for recognition
    for (a, b, c, d) in faces:
        #print(a, b, c, d)   

        # Region of Interest (ROI)
        roi_grey = grey[ b : b + d, a : a + c]
        roi_colour = record[ b : b + d, a : a + c]

        
        # Defining of recognized faces as ROI
        num, conf = face_identification.predict(roi_grey)

        
        # Actual pixel neighbourghs sensitivity
        if conf >= 45 and conf <= 85:            
            print(num)
            print(folder_name[num])

            # Display of persons name in the recording
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = folder_name[num]
            colour = (0, 255, 255)
            thickness = 2
            cv2.putText(record, name, (a,b-20), font, 1, colour, thickness, cv2.LINE_AA) 
            cv2.putText(grey , name, (a,b-20), font, 1, colour, thickness, cv2.LINE_AA)

        #  save ROI as .png
        your_face = "your_face.png"
        cv2.imwrite( your_face, roi_grey)


    # Defining rectangle around face
        colour = (0 , 255, 0) #BGR
        thickness = 2
        wideness = a + c
        height = b + d

        # Face rectangle
        cv2.rectangle( record, (a, b), (wideness, height), colour, thickness)
        cv2.rectangle( grey, (a, b), (wideness, height), (0, 255, 0), thickness)

        # eyes rectangles
        eyes = eye_recognition.detectMultiScale(roi_colour, scaleFactor = 2, minNeighbors = 5)
        for (e, f, g, h) in eyes:
            cv2.rectangle(roi_colour, (e, f), (e + g, f + h), (255, 0, 0), 1)
            cv2.rectangle(roi_grey, (e, f), (e + g, f + h), (255, 0, 0), 1)

        # Save eyes as .png
        roi_grey2 = grey[f : f + h, e : e + g]
        your_eye = "your_eye.png"
        cv2.imwrite( your_eye, roi_grey2)

    # Header of displayed record
    cv2.imshow('Philips Camera', record)   
    cv2.imshow('Grayscales', grey)

    # Quit program with "q"
    if cv2.waitKey(20) & 0xFF == ord('q'): 
        print('You quit the program. Come back soon :)')
        break

# closing of recording
camera.release()
cv2.destroyAllWindows()
