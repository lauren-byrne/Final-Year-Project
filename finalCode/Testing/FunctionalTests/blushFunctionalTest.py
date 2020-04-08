import cv2
import numpy as np
import dlib
from imutils import face_utils


def detection(faces, image):
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(image, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        landmarks2 = face_utils.shape_to_np(landmarks)

        crop = image[landmarks2[29][1]:landmarks2[33][1], landmarks2[54][0]:landmarks2[12][0]]  # right cheeks
        crop2 = image[landmarks2[29][1]:landmarks2[33][1], landmarks2[4][0]:landmarks2[48][0]]  # left cheek

        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('right cheek', crop)
        # cv2.imshow('left cheek', crop2)

        cheek_average = (np.mean(crop) + np.mean(crop2))
        return cheek_average


no_blush = cv2.imread('nob.png')
blush = cv2.imread('b.png')
no_blush = cv2.resize(no_blush, (0, 0), fx=0.7, fy=0.7)
blush = cv2.resize(blush, (0, 0), fx=0.7, fy=0.7)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

no_blush_gray = cv2.cvtColor(no_blush, cv2.COLOR_BGR2GRAY)
blush_gray = cv2.cvtColor(blush, cv2.COLOR_BGR2GRAY)

no_blush_equalised = cv2.equalizeHist(no_blush_gray)
blush_equalised = cv2.equalizeHist(blush_gray)

faces_no_blush = detector(no_blush_equalised)
faces_blush = detector(blush_equalised)

no_blush_average = detection(faces_no_blush, no_blush_equalised)
blush_average = detection(faces_blush, blush_equalised)

difference = ((no_blush_average - blush_average)/no_blush_average)*100

print('difference: ', difference)

cv2.imshow('no blush', no_blush)
cv2.imshow('blush', blush)

key = cv2.waitKey(0)
# 27 is ESC key

