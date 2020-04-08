import cv2
import dlib
import math
import pointMaths
from imutils import face_utils
from blinkTracking import get_blinking_ratio

image = cv2.imread('no_blink.jpg')

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

left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

# Track blinking
blinking_ratio = (right_eye_ratio + left_eye_ratio) / 2

if blinking_ratio > 4.5:
    print('blink')
else:
    print('no blink')

key = cv2.waitKey(0)

