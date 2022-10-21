import os
import sys
import cv2
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt


lables = [0,1]
lablesName = ['Messi','Ronaldo']
training_data_folder_path = 'Training'
test_data_folder_path = 'Test'
haarcascade_frontalface = 'haarcascade_frontalface_alt.xml'


detected_faces, face_labels = prepare_training_data("Training")

eigenfaces_recognizer = cv2.face.EigenFaceRecognizer_create()

eigenfaces_recognizer.train(detected_faces, np.array(face_labels))


def detect_all_face(input_img):
    detect_faces  = []

    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
    faces = face_cascade.detectMultiScale(input_img, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
    for face in faces:
        (x, y, w, h) = face
        detect_face = image[y:y+w, x:x+h], face
        detect_faces.append(detect_face)
    return detect_faces


def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]


def prepare_training_data(training_data_folder_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(training_data_folder_path)
    for dir_name in traning_image_dirs:
        training_image_path = training_data_folder_path + "/" + dir_name
        training_images_names = os.listdir(training_image_path)
        
        for image_name in training_images_names:
            image_path = training_image_path  + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)

            if face is not -1:
                resized_face = cv2.resize(face, (121,121), interpolation = cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(lables[lablesName.index(dir_name)])

    return detected_faces, face_labels


def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(test_image, label_text, x, y, h):
    cv2.putText(test_image, label_text, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def predict(test_image):
    detected_faces = detect_all_face(test_image)

    for detected_face in detected_faces:
        if (detect_face == -1):
           continue
        resized_test_image = cv2.resize(detected_face[0], (121,121), interpolation = cv2.INTER_AREA)
        #label = eigenfaces_recognizer.predict(resized_test_image)
        #label_text = lablesName[lables.index(label[0])]

        draw_rectangle(test_image, detected_face[1])
        print(detected_face[1])
        #draw_text(test_image, label_text, detected_face[1][0], detected_face[1][1], detected_face[1][2] + 20)

    return test_image


test_image = cv2.imread(r"C:\Users\phanm\Downloads\tt.jpg")
predicted_image = predict(test_image)

cv2.imshow('1', predicted_image)
cv2.waitKey()

