'''
Lauren Byrne C16452764 FYP - Prototype
Program that will detect a face from webcam input using Dlib HoG face detection.
Detected face will have a green bounding box drawn around it.
Using Dlib facial landmark detection, facial landmarks will be detected
and drawn on face using blue circles

----References----
Face landmarks detection - Opencv with Python - YouTube [Internet]. [cited 2020 Jan 15].
Available from: https://www.youtube.com/watch?v=MrRGVOhARYY&t=429s

Eye Blinking detection – Gaze controlled keyboard with Python and Opencv p.2 - YouTube [Internet].
[cited 2020 Jan 15]. Available from: https://www.youtube.com/watch?v=mMObcjHs59E&t=182s
Eye Blinking detection – Gaze controlled keyboard with Python and Opencv p.2 - YouTube [Internet]. [cited 2020 Jan 15].
Available from: https://www.youtube.com/watch?v=mMObcjHs59E&t=182s

Eye motion tracking - Opencv with Python - YouTube [Internet].
[cited 2019 Nov 6]. Available from: https://www.youtube.com/watch?v=kbdbZFT9NQI&t=1166s

'''

import cv2
import dlib
import csv
from math import hypot
import numpy as np


cap = cv2.VideoCapture(0)

#dlib face detector and facial landmark detector models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_points = list()


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # using eye points for drawing
   # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 1)
   # ver_line = cv2.line(frame, center_top, center_bottom, (0, 0, 255), 1)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print('frame', frame)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        # can change to get(5) - will get
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)


        print(landmarks)

        # mapping each facial landmark
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        RightEyeX = landmarks.part(46).x
        RightEyeY = landmarks.part(46).y

        #Track blinking
        blinking_ratio = (right_eye_ratio + left_eye_ratio)/2
        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 3, (255, 0, 0))


        #Track gaze
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        cv2.polylines(frame, [left_eye_region], True, (0, 150, 255), 2)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        eye = frame[min_y: max_y, min_x: max_x]
        eye = cv2.resize(eye, None, fx=5, fy=5)

        #print(eye)

        cv2.imshow('Eye', eye)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()

    # save on pressing 'y'
'''    if cv2.waitKey(1) & 0xFF == ord('y'):
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # print('x :', x)
            # print('y :', y)
            face_points.append(x)
            face_points.append(y)

        with open('test.csv', 'a', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(face_points)

        face_points = []

    if cv2.waitKey(10) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
cap.release()
cv2.destroyAllWindows()
# print('landmarks:  ', face_points)'''


