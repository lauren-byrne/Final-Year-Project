import numpy as np
import cv2
from imutils import face_utils


def get_blush_change(frame, cheek_list, cheek_points, facial_landmarks):
    landmarks2 = face_utils.shape_to_np(facial_landmarks)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)

    crop = frame[landmarks2[cheek_points[0]][cheek_points[1]]:landmarks2[cheek_points[2]][cheek_points[1]],
           landmarks2[cheek_points[4]][cheek_points[5]]:landmarks2[cheek_points[6]][cheek_points[5]]]  # right cheeks
    crop2 = frame[landmarks2[cheek_points[0]][cheek_points[1]]:landmarks2[cheek_points[2]][cheek_points[1]],
            landmarks2[cheek_points[7]][cheek_points[5]]:landmarks2[cheek_points[8]][cheek_points[5]]]  # left cheek


    cv2.imshow('right cheek', crop)
    cv2.imshow('left cheek', crop2)

    cheek_average = (np.mean(crop) + np.mean(crop2))

    cheek_list.append(cheek_average)


    return cheek_average
