import cv2
import dlib
import numpy as np
from imutils import face_utils
from gazeTracking import get_white_ratio
from gazeTracking import get_eye_shape

image = cv2.imread('right.jpg')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    landmarks = predictor(image, face)

    x = landmarks.part(20).x
    y = landmarks.part(20).y

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    landmarks2 = face_utils.shape_to_np(landmarks)

height = np.size(image, 0)
width = np.size(image, 1)

black_img = np.zeros((height, width), dtype="uint8")

# calling functions to retrieve the eye shape of both the right and left eye to appear on new black frame
min_x_left, max_x_left, min_y_left, max_y_left, black_img = get_eye_shape([36, 37, 38, 39, 40, 41], landmarks,
                                                                          black_img)
min_x_right, max_x_right, min_y_right, max_y_right, black_img = get_eye_shape([42, 43, 44, 45, 46, 47],
                                                                              landmarks, black_img)

left_ratio = get_white_ratio(min_x_left, max_x_left, min_y_left, max_y_left, black_img, image)
right_ratio = get_white_ratio(min_x_right, max_x_right, min_y_right, max_y_right, black_img, image)

gaze_ratio = (left_ratio + right_ratio) / 2

if gaze_ratio < 0.92:
    print('left')
elif gaze_ratio > 0.98:
   print('right')
else:
    print('center')

print(gaze_ratio)

key = cv2.waitKey(0)

