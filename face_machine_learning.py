# Machine learning program using the folder of pictures with faces of persons inside
# Editor: Philip Schmidt
# Date: 24.01.2020
# Detailed script documentation in form of comments right next to the code

# Enjoy using the program!

# Installing and upgrading all necessary packages
import os, sys
from subprocess import call

my_packages = ['opencv-contrib-python', 'pillow', 'pickle']
def upgrade(package_list):
    call(['pip', 'install', '--upgrade', '--user'] + package_list)
upgrade(my_packages)

import cv2
import os
import numpy as np 
from PIL import Image
import pickle

main_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(main_path, "pictures")

face_recognition = cv2.CascadeClassifier('haarcascades\\haarcascade_frontalface_alt2.xml') # xml. data for face_recognition

face_identification = cv2.face.LBPHFaceRecognizer_create() # identifying with local binary patterns histograms from openCV

y_folder_name = []                   # creating of lists
x_training = []

add_id = 0
folder_ids = {}

# Defining of all adresses and paths from the folder pictures
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join( root, file)

            # defining of the folder_name 
            folder_name = os.path.basename(os.path.dirname(path)).replace( " ", "-").lower()
            
            
            if not folder_name in folder_ids:               
                folder_ids[folder_name] = add_id
                add_id += 1
            new_id = folder_ids[folder_name]
           

            # y_folder_name.append(folder_name)
            # x_training.append(path)

            gray_image = Image.open(path).convert("L") # Konversion in gray scales, pil = python image library
            size = (550, 550)
            resized_image = gray_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(resized_image, "uint8") # conversion into numpy-array
            #print(image_array)

            # face_recognition
            faces = face_recognition.detectMultiScale( image_array, scaleFactor = 2, minNeighbors = 4)

            for ( a, b, c, d) in faces:
                roi = image_array[ b : b + d, a : a + c]

                x_training.append(roi)
                y_folder_name.append(new_id)

# output of all foldernames and calculated matrices
print(folder_name, path)
print(folder_ids)
print(image_array)
print(y_folder_name)
print(x_training)


# picking folder (with pickle)
with open("folder_name.pickle", 'wb') as f:  
    pickle.dump(folder_ids, f)

# Training for identification
face_identification.train(x_training, np.array(y_folder_name))   # actual machine learning with x_training of ROI and numpy array from folder_name ,training
face_identification.save("facetrainer.yml")                      # training