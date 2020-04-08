import numpy as np
import cv2


# function that return the extreme points of the outline of an eye
# will also draw outline on new black frame
def get_eye_shape(eye_points, facial_landmarks, black_image):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                          np.int32)

    # calculating the extreme points
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    cv2.polylines(black_image, [eye_region], True, 255, 2)
    cv2.fillPoly(black_image, [eye_region], 255)

    return min_x, max_x, min_y, max_y, black_image


def get_white_ratio(min_x, max_x, min_y, max_y, black_img, frame):
    masked_eye = cv2.bitwise_and(frame, frame, mask=black_img)

    # cropping the eye at the extreme points from the masked image - any skin around eye will now be black
    eye = masked_eye[min_y: max_y, min_x: max_x]
    eye = cv2.resize(eye, None, fx=5, fy=5)

    # creating threshold for eye
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    T, B = cv2.threshold(gray_eye, 50, maxval=255, type=cv2.THRESH_BINARY)

    eye_height = np.size(B, 0)
    eye_width = np.size(B, 1)

    # splitting the eye in left and right sides
    left_white_eye = gray_eye[0:eye_height, 0:int(eye_width / 2)]
    right_white_eye = gray_eye[0:eye_height, int(eye_width / 2):eye_width]

    # counting the amount of black (pupil and iris) that appears in each side of each eye
    left_white_amount = cv2.countNonZero(left_white_eye)
    right_white_amount = cv2.countNonZero(right_white_eye)

    white_ratio = left_white_amount / right_white_amount

    return white_ratio




