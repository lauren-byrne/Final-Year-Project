import numpy as np
import cv2
from imutils import face_utils


def get_blush_change(frame, cheek_list, cheek_points, facial_landmarks):


    landmarks2 = face_utils.shape_to_np(facial_landmarks)

    crop = frame[landmarks2[cheek_points[0]][cheek_points[1]]:landmarks2[cheek_points[2]][cheek_points[1]], landmarks2[cheek_points[4]][cheek_points[5]]:landmarks2[cheek_points[6]][cheek_points[5]]]  # right cheeks
    crop2 = frame[landmarks2[cheek_points[0]][cheek_points[1]]:landmarks2[cheek_points[2]][cheek_points[1]], landmarks2[cheek_points[7]][cheek_points[5]]:landmarks2[cheek_points[8]][cheek_points[5]]]  # left cheek

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    cv2.imshow('right cheek', crop)
    cv2.imshow('left cheek', crop2)

    cheek_average = (np.mean(crop) + np.mean(crop2))

    cheek_list.append(cheek_average)
    cheek_array = mat = np.array(cheek_list)
    # print('cheek list: ', cheek_list)

    return cheek_average