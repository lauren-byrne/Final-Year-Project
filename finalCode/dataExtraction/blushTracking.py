import numpy as np
import cv2
from imutils import face_utils


# function will return the average pixel value for the cheek region
def get_blush_change(frame, cheek_points, facial_landmarks):
    landmarks = face_utils.shape_to_np(facial_landmarks)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)

    # right cheeks
    right_crop = frame[landmarks[cheek_points[0]][cheek_points[1]]:landmarks[cheek_points[2]][cheek_points[1]],
                 landmarks[cheek_points[4]][cheek_points[5]]:landmarks[cheek_points[6]][cheek_points[5]]]

    # left cheek
    left_crop = frame[landmarks[cheek_points[0]][cheek_points[1]]:landmarks[cheek_points[2]][cheek_points[1]],
                landmarks[cheek_points[7]][cheek_points[5]]:landmarks[cheek_points[8]][cheek_points[5]]]

    # cv2.imshow('right cheek', right_crop)
    # cv2.imshow('left cheek', left_crop)

    cheek_average = (np.mean(right_crop) + np.mean(left_crop))

    return cheek_average
